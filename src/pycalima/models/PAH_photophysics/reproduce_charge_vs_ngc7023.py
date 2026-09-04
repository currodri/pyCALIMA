"""
reproduce_charge_vs_ngc7023.py
==============================
Equilibrium PAH charge-state distribution f(Z) along the NW PDR of NGC7023,
using physically consistent ne(Av), nH(Av), T(Av), and G0(Av) profiles
digitised from Andrews et al. (2016).

Instead of scanning a synthetic gamma grid at fixed gas conditions, each point
is evaluated at the actual (G0, ne, nH, T) found at that cloud depth, then
plotted against the local ionisation parameter

    gamma(Av) = G0(Av) * sqrt(T(Av)) / ne(Av)

so the resulting curve traces the real NGC7023 NW PDR track in gamma-space.

Both the calibrated Andrews+16 se prescription and the full WR se (with
Cagliari alpha and Li&Draine kr) are shown.

Usage
-----
    python -m models.PAH_photophysics.reproduce_charge_vs_ngc7023
"""

from __future__ import annotations

import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from pycalima.models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    recombination_rate_Bakes1994,
    attachment_rate_Bakes1994,
    se_neutral_Andrews2016,
    se_neutral_WR_full,
)
from pycalima.models.PAH_photophysics.pah_dissociation import compute_total_photoionisation_rate
from pycalima.models.PAH_photophysics.pah_network_solver import PAHNetworkSolver
from pycalima.models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_u_E, load_kurucz_I_nu
from pycalima.models.PAH_photophysics.reproduce_andrews16_fig9 import (
    PAH_DEFS as _PAH_DEFS_BASE, _PAH_ORDER, _STATES_DIR, _HV_EV,
    _build_kdis, _worker,
)

# ── Add electron affinities ───────────────────────────────────────────────────
_EA = {'C24': 0.47, 'C54': 1.44, 'C96': 3.11}
PAH_DEFS = {k: dict(**v, EA=_EA[k]) for k, v in _PAH_DEFS_BASE.items()}

# ── PDR data directory ────────────────────────────────────────────────────────
_PDR_DIR = ROOT / 'external_data' / 'NWPDR_NGC7023'

N_AV = 25   # number of Av evaluation points


def _load_profile(fname: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(_PDR_DIR / fname, delimiter=',')
    return d[:, 0], d[:, 1]


def _make_interp(x, y):
    return interp1d(x, y, kind='linear', bounds_error=False,
                    fill_value=(y[0], y[-1]))


def load_pdr_profiles(n_pts: int = N_AV):
    """
    Load all PDR profiles and interpolate to a common Av grid.

    nH.csv  = n(H) atomic hydrogen density  [cm^-3]  (NOT total H nuclei)
    nH2.csv = n(H2) molecular hydrogen density [cm^-3]
    G0nH.csv = G0 / n(H)  [cm^3]  → G0 = G0nH × n(H)

    Returns
    -------
    av_grid : (N,) array  — visual extinction [mag]
    G0      : (N,) array  — radiation field intensity [Habing units]
    ne      : (N,) array  — electron density [cm^-3]
    nH      : (N,) array  — atomic H density n(H) [cm^-3]
    T       : (N,) array  — gas temperature [K]
    gamma   : (N,) array  — ionisation parameter G0*sqrt(T)/ne (recomputed)
    """
    av_g0nH, g0nH  = _load_profile('G0nH.csv')
    av_ne,   ne    = _load_profile('ne.csv')
    av_nH,   nH    = _load_profile('nH.csv')    # n(H) atomic only
    av_nH2,  nH2   = _load_profile('nH2.csv')   # n(H2) molecular
    av_T,    T     = _load_profile('temperature.csv')

    av_min = max(av_g0nH.min(), av_ne.min(), av_nH.min(), av_nH2.min(), av_T.min())
    av_max = min(av_g0nH.max(), av_ne.max(), av_nH.max(), av_nH2.max(), av_T.max())

    av_grid = np.linspace(av_min, av_max, n_pts)

    G0nH_f = _make_interp(av_g0nH, g0nH)
    ne_f   = _make_interp(av_ne,   ne)
    nH_f   = _make_interp(av_nH,   nH)
    nH2_f  = _make_interp(av_nH2,  nH2)
    T_f    = _make_interp(av_T,    T)

    nH_grid    = nH_f(av_grid)                      # n(H) atomic
    nH2_grid   = nH2_f(av_grid)                     # n(H2) molecular
    nH_tot_grid = nH_grid + 2.0 * nH2_grid          # total H nuclei density
    ne_grid    = ne_f(av_grid)
    T_grid     = T_f(av_grid)
    G0_grid    = G0nH_f(av_grid) * nH_grid          # G0 = (G0/n(H)) × n(H)
    gamma_grid = G0_grid * np.sqrt(T_grid) / ne_grid

    return av_grid, G0_grid, ne_grid, nH_grid, T_grid, gamma_grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading PDR profiles ...", flush=True)
    av_grid, G0_grid, ne_grid, nH_grid, T_grid, gamma_grid = load_pdr_profiles()

    print(f"  Av range: [{av_grid[0]:.3f}, {av_grid[-1]:.3f}] mag  ({N_AV} pts)")
    print(f"  G0  range: [{G0_grid.min():.2e}, {G0_grid.max():.2e}]")
    print(f"  ne  range: [{ne_grid.min():.2e}, {ne_grid.max():.2e}] cm^-3")
    print(f"  n(H) range: [{nH_grid.min():.2e}, {nH_grid.max():.2e}] cm^-3  (atomic H only)")
    print(f"  T   range: [{T_grid.min():.1f}, {T_grid.max():.1f}] K")
    print(f"  gamma range: [{gamma_grid.min():.2e}, {gamma_grid.max():.2e}]")

    print("\nComputing Kurucz G0 base ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    print(f"  G0_base = {G0_base:.4e}", flush=True)

    # Cross-section tables
    xsect_tables: dict[str, np.ndarray] = {}
    for name, pdef in PAH_DEFS.items():
        a0 = afromNc(pdef['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0)
        E_cs = _HV_EV / w
        idx  = np.argsort(E_cs)
        xsect_tables[name] = np.column_stack([E_cs[idx], C_abs[idx]])

    # Build worker task list: one per (PAH, Av point)
    tasks = []
    for name, pdef in PAH_DEFS.items():
        modes_path = str(_STATES_DIR / pdef['modes'])
        xsect      = xsect_tables[name]
        for iav in range(N_AV):
            G0 = float(G0_grid[iav])
            tasks.append((name, modes_path, xsect, G0, G0_base,
                          pdef['IP1'], pdef['IP2']))

    n_tasks   = len(tasks)
    n_workers = max(1, min(cpu_count() - 1, n_tasks))
    print(f"\nRunning {n_tasks} rate-table tasks on {n_workers} workers ...", flush=True)

    with Pool(n_workers) as pool:
        raw_results = pool.map(_worker, tasks)

    # pool.map preserves order: results come back (PAH0_iav0, PAH0_iav1, ...,
    # PAH1_iav0, ...) matching the task ordering above.
    rates_table: dict[str, dict] = {name: {} for name in PAH_DEFS}
    result_idx = 0
    for name in PAH_DEFS:
        for iav in range(N_AV):
            _pah_name, _G0_val, rates = raw_results[result_idx]
            rates_table[name][iav] = rates
            result_idx += 1

    print("Rate tables complete.\n", flush=True)

    # Photodetachment reference rates (k_det ∝ G0)
    print("Computing photodetachment reference rates ...", flush=True)
    kurucz_I = load_kurucz_I_nu(15000)
    def _field_ref(nu): return kurucz_I(nu) / G0_base
    k_det_ref: dict[str, float] = {}
    for name, pdef in PAH_DEFS.items():
        a0 = afromNc(pdef['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0)
        E_cs = _HV_EV / w; idx = np.argsort(E_cs)
        xsect = np.column_stack([E_cs[idx], C_abs[idx]])
        k_det_ref[name] = float(compute_total_photoionisation_rate(_field_ref, xsect, IP=pdef['EA']))
        print(f"  {name}: k_det_ref={k_det_ref[name]:.3e}  (EA={pdef['EA']} eV)", flush=True)

    # ── Solve charge network at each Av point ─────────────────────────────────
    # fZ_*[name] shape: (N_AV, 4) — f(Z=-1,0,+1,+2)
    fZ_calib: dict[str, np.ndarray] = {}
    fZ_wr:    dict[str, np.ndarray] = {}

    for name, pdef in PAH_DEFS.items():
        Nc, Nh0  = pdef['Nc'], pdef['Nh0']
        solo, duo = pdef['solo'], pdef['duo']

        solver = PAHNetworkSolver(Nc=Nc, Nh0=Nh0, parent_solo=solo, parent_duo=duo)
        a_cm   = afromNc(Nc)

        # se_calib is T-independent for Andrews+16 prescription
        se_c = se_neutral_Andrews2016(Nc, pdef['EA'])
        print(f"\nSolving network for {name}  ({solver.N} species)  se_calib={se_c:.4f}", flush=True)

        fZ_c = np.zeros((N_AV, 4))
        fZ_w = np.zeros((N_AV, 4))

        for iav in range(N_AV):
            G0 = G0_grid[iav]
            ne = ne_grid[iav]
            nH = nH_grid[iav]
            T  = T_grid[iav]

            n_H = nH     # nH.csv is already n(H) atomic hydrogen
            n_e = ne

            # se_WR depends on T
            se_w = se_neutral_WR_full(Nc, pdef['EA'], T_K=T)

            k_rec1_coeff = recombination_rate_Bakes1994(Nc, Z=1, se=1.0, T=T, ne=1.0)
            k_rec2_coeff = recombination_rate_Bakes1994(Nc, Z=2, se=1.0, T=T, ne=1.0)
            k_rec = np.array([0.0, 0.0, k_rec1_coeff, k_rec2_coeff])

            k_att_c = attachment_rate_Bakes1994(Nc, se=se_c, T=T, ne=1.0)
            k_att_w = attachment_rate_Bakes1994(Nc, se=se_w, T=T, ne=1.0)

            rates = rates_table[name][iav]
            k_Hloss, k_H2loss = _build_kdis(solver, rates, solo, duo)
            k_ion = np.array([
                k_det_ref[name] * G0,
                rates['k_ion_1'],
                rates['k_ion_2'],
                0.0,
            ])

            def _solve(k_att_coeff):
                try:
                    n2d = solver.solve_equilibrium(
                        k_Hloss, k_H2loss, k_ion, k_rec, k_att_coeff,
                        n_H=n_H, n_e=n_e, T=T, a_pah_cm=a_cm,
                        n_PAH_total=1.0, method='newton',
                    )
                except RuntimeError:
                    n2d = solver.solve_equilibrium(
                        k_Hloss, k_H2loss, k_ion, k_rec, k_att_coeff,
                        n_H=n_H, n_e=n_e, T=T, a_pah_cm=a_cm,
                        n_PAH_total=1.0, method='direct',
                    )
                n_tot = n2d.sum()
                row = np.zeros(4)
                if n_tot > 0:
                    for iz in range(4):
                        row[iz] = n2d[iz, :].sum() / n_tot
                return row

            fZ_c[iav] = _solve(k_att_c)
            fZ_w[iav] = _solve(k_att_w)

            if iav % 5 == 0:
                print(f"  Av={av_grid[iav]:.2f}  G0={G0:.1e}  T={T:.0f}K  "
                      f"se_WR={se_w:.2e}  fZ(Z=-1)_c={fZ_c[iav,0]:.3e}", flush=True)

        fZ_calib[name] = fZ_c
        fZ_wr[name]    = fZ_w

    _plot(av_grid, gamma_grid, fZ_calib, fZ_wr, G0_grid, T_grid, ne_grid)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _load_andrews(pah_key: str) -> dict:
    ext = ROOT / 'external_data'
    data = {}
    for Z, tag in [(-1, 'anion'), (0, 'neutral'), (1, 'cation'), (2, 'dication')]:
        f = ext / f'{pah_key}HN_{tag}_andrews16.csv'
        if f.exists():
            data[Z] = np.loadtxt(f, delimiter=',')
    return data


def _plot(av_grid, gamma_grid, fZ_calib, fZ_wr, G0_grid, T_grid, ne_grid) -> None:
    Z_colors = {-1: '#9467bd', 0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    Z_labels = {-1: r'$Z=-1$',  0: r'$Z=0$',  1: r'$Z=+1$',  2: r'$Z=+2$'}
    Z_ls     = {-1: '--',       0: '-',        1: '-',         2: '-.'}
    Z_markers= {-1: 'D',        0: 'o',        1: 's',         2: '^'}

    andrews = {name: _load_andrews(name) for name in _PAH_ORDER}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)

    for col, name in enumerate(_PAH_ORDER):
        ax   = axes[col]
        pdef = PAH_DEFS[name]

        for iz, Z in enumerate([-1, 0, 1, 2]):
            lbl_c = (Z_labels[Z] + ' (calib)') if col == 0 else None
            lbl_w = (Z_labels[Z] + ' (WR)')    if col == 0 else None
            ax.plot(gamma_grid, fZ_calib[name][:, iz],
                    color=Z_colors[Z], ls=Z_ls[Z], lw=1.5, alpha=0.45, label=lbl_c)
            ax.plot(gamma_grid, fZ_wr[name][:, iz],
                    color=Z_colors[Z], ls=Z_ls[Z], lw=2.2, label=lbl_w)
            # Mark each Av point so the direction of travel is visible
            ax.scatter(gamma_grid, fZ_wr[name][:, iz],
                       color=Z_colors[Z], marker=Z_markers[Z], s=18, zorder=4,
                       linewidths=0.0, alpha=0.6)

        # Overlay Andrews+16 digitised synthetic curves
        for Z, arr in andrews[name].items():
            lbl = f'Andrews+16 $Z={Z:+d}$' if col == 0 else None
            ax.scatter(arr[:, 0], arr[:, 1],
                       color=Z_colors[Z], marker='o', s=30, zorder=5,
                       edgecolors='k', linewidths=0.4, label=lbl)

        ax.set_xscale('log')
        ax.set_xlim(5e0, 5e4)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r'$\gamma = G_0\,\sqrt{T}\,/\,n_e$', fontsize=11)
        ax.set_title(pdef['label'], fontsize=11)
        if col == 0:
            ax.set_ylabel('Charge-state fraction', fontsize=11)
            ax.legend(fontsize=7.5, ncol=2)
        ax.tick_params(labelsize=9)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)

    fig.suptitle(
        r'PAH charge distribution — NGC7023 NW PDR profile'
        '\n'
        r'Each point = one depth $A_V$; $\gamma(A_V) = G_0\sqrt{T}/n_e$ self-consistent'
        '\n'
        r'Thick/markers: WR $s_e$ (Cagliari $\alpha$, Li\&Draine $k_r$, $T=T(A_V)$)'
        r'   Faded: calibrated $s_e$ (Andrews+16)'
        r'   Circles: Andrews+16 Fig. 8 [digitised]',
        fontsize=9.5,
    )

    out = ROOT / 'pah_charge_vs_ngc7023_pdr.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
