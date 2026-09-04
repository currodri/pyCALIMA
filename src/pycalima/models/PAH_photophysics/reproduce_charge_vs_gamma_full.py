"""
reproduce_charge_vs_gamma_full.py
==================================
Equilibrium charge-state distribution f(Z), marginalized over ALL
hydrogenation states — analogous to Andrews et al. (2016) Fig. 8.

For each G0, the full (Z, Nh) network is solved at fixed nH.  The charge
fractions are obtained by summing over all Nh:

    f(Z) = Σ_Nh n(Z, Nh) / Σ_{Z,Nh} n(Z, Nh)

This differs from reproduce_charge_vs_gamma.py (which fixes Nh = Nh0): here
the hydrogenation-state distribution is self-consistent with the radiation
field and gas density, so dehydrogenated and super-H populations contribute
to the effective charge balance.

Three panels (one per PAH species) in one figure.  The x-axis is the
standard ionisation parameter γ = G0 √T / ne.

Usage
-----
    python -m models.PAH_photophysics.reproduce_charge_vs_gamma_full
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    recombination_rate_Bakes1994,
    attachment_rate_Bakes1994,
    se_neutral_Andrews2016,
    se_neutral_WR_full,
)
from models.PAH_photophysics.pah_dissociation import compute_total_photoionisation_rate
from models.PAH_photophysics.pah_network_solver import PAHNetworkSolver
from models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from models.PAH_photophysics.pah_radiation import load_kurucz_u_E, load_kurucz_I_nu

# Reuse PAH definitions, RRKM classes, worker, and helpers from Fig. 9
from models.PAH_photophysics.reproduce_andrews16_fig9 import (
    PAH_DEFS as _PAH_DEFS_BASE, _PAH_ORDER, _STATES_DIR, _HV_EV,
    _build_kdis, _worker,
    F_H2, X_E, T_GAS,
)

# Add electron affinities (used for correct photodetachment threshold)
_EA = {'C24': 0.47, 'C54': 1.44, 'C96': 3.11}   # eV  (C96 corrected: BT94 EA(1)=4.4-0.5×25.1/√96)
PAH_DEFS = {k: dict(**v, EA=_EA[k]) for k, v in _PAH_DEFS_BASE.items()}

# ─── Grid ─────────────────────────────────────────────────────────────────────
NG0     = 50
G0_GRID = np.logspace(-3, 5, NG0)   # wide: γ ~ 1 – 1e8 at nH=100

# Fixed gas density
NH_REF = 100.0   # cm^-3


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Computing Kurucz G0 base ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    print(f"  G0_base = {G0_base:.4e}", flush=True)

    # Cross-section tables (one per PAH size)
    xsect_tables: dict[str, np.ndarray] = {}
    for name, pdef in PAH_DEFS.items():
        a0 = afromNc(pdef['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0)
        E_cs = _HV_EV / w
        idx  = np.argsort(E_cs)
        xsect_tables[name] = np.column_stack([E_cs[idx], C_abs[idx]])

    # Build task list (same structure as Fig. 9 worker)
    tasks = []
    for name, pdef in PAH_DEFS.items():
        modes_path = str(_STATES_DIR / pdef['modes'])
        xsect      = xsect_tables[name]
        for G0 in G0_GRID:
            tasks.append((name, modes_path, xsect, float(G0), G0_base,
                          pdef['IP1'], pdef['IP2']))

    n_tasks   = len(tasks)
    n_workers = max(1, min(cpu_count() - 1, n_tasks))
    print(f"\nRunning {n_tasks} rate-table tasks on {n_workers} workers ...", flush=True)

    with Pool(n_workers) as pool:
        raw_results = pool.map(_worker, tasks)

    rates_table: dict[str, dict] = {name: {} for name in PAH_DEFS}
    for pah_name, G0, rates in raw_results:
        ig0 = int(np.argmin(np.abs(G0_GRID - G0)))
        rates_table[pah_name][ig0] = rates

    print("Rate tables complete.\n", flush=True)

    # Physical conditions
    nH_total = NH_REF
    n_H      = (1.0 - F_H2) * nH_total   # atomic H density
    n_e      = X_E * nH_total             # electron density

    # γ axis: one value per G0
    gamma_grid = G0_GRID * np.sqrt(T_GAS) / n_e

    # Pre-compute k_det_ref (photodetachment at G0=1 using EA threshold)
    # k_det scales linearly with G0, so k_det(G0) = k_det_ref * G0
    print("Computing photodetachment reference rates (EA threshold) ...", flush=True)
    kurucz_I = load_kurucz_I_nu(15000)
    def _field_ref(nu): return kurucz_I(nu) / float(compute_base_g0(load_kurucz_u_E(15000)))
    k_det_ref: dict[str, float] = {}
    for name, pdef in PAH_DEFS.items():
        a0 = afromNc(pdef['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0)
        E_cs = _HV_EV / w; idx = np.argsort(E_cs)
        xsect = np.column_stack([E_cs[idx], C_abs[idx]])
        k_det_ref[name] = float(compute_total_photoionisation_rate(_field_ref, xsect, IP=pdef['EA']))
        print(f"  {name}: k_det_ref={k_det_ref[name]:.3e}  (EA={pdef['EA']} eV)", flush=True)

    # ── Solve full network, marginalize over Nh ──
    # fZ_*[name] shape: (NG0, 4)  — f(Z=-1,0,+1,+2)
    fZ_calib: dict[str, np.ndarray] = {}
    fZ_wr:    dict[str, np.ndarray] = {}

    for name, pdef in PAH_DEFS.items():
        Nc, Nh0  = pdef['Nc'], pdef['Nh0']
        solo, duo = pdef['solo'], pdef['duo']

        solver = PAHNetworkSolver(Nc=Nc, Nh0=Nh0, parent_solo=solo, parent_duo=duo)
        a_cm   = afromNc(Nc)

        k_rec1_coeff = recombination_rate_Bakes1994(Nc, Z=1, se=1.0, T=T_GAS, ne=1.0)
        k_rec2_coeff = recombination_rate_Bakes1994(Nc, Z=2, se=1.0, T=T_GAS, ne=1.0)
        k_rec = np.array([0.0, 0.0, k_rec1_coeff, k_rec2_coeff])

        se_c  = se_neutral_Andrews2016(Nc, pdef['EA'])
        se_w  = se_neutral_WR_full(Nc, pdef['EA'], T_K=T_GAS)
        k_att_c = attachment_rate_Bakes1994(Nc, se=se_c, T=T_GAS, ne=1.0)
        k_att_w = attachment_rate_Bakes1994(Nc, se=se_w, T=T_GAS, ne=1.0)
        print(f"Solving network for {name}  ({solver.N} species)  "
              f"se_calib={se_c:.4f}  se_WR_full={se_w:.3e}", flush=True)

        fZ_c = np.zeros((NG0, 4))
        fZ_w = np.zeros((NG0, 4))

        def _solve_g0(ig0, k_att_coeff):
            rates = rates_table[name][ig0]
            k_Hloss, k_H2loss = _build_kdis(solver, rates, solo, duo)
            G0 = G0_GRID[ig0]
            k_ion = np.array([
                k_det_ref[name] * G0,
                rates['k_ion_1'],
                rates['k_ion_2'],
                0.0,
            ])
            try:
                n2d = solver.solve_equilibrium(
                    k_Hloss, k_H2loss, k_ion, k_rec, k_att_coeff,
                    n_H=n_H, n_e=n_e, T=T_GAS, a_pah_cm=a_cm,
                    n_PAH_total=1.0, method='newton',
                )
            except RuntimeError:
                n2d = solver.solve_equilibrium(
                    k_Hloss, k_H2loss, k_ion, k_rec, k_att_coeff,
                    n_H=n_H, n_e=n_e, T=T_GAS, a_pah_cm=a_cm,
                    n_PAH_total=1.0, method='direct',
                )
            n_tot = n2d.sum()
            row = np.zeros(4)
            if n_tot > 0:
                for iz in range(4):
                    row[iz] = n2d[iz, :].sum() / n_tot
            return row

        for ig0 in range(NG0):
            fZ_c[ig0] = _solve_g0(ig0, k_att_c)
            fZ_w[ig0] = _solve_g0(ig0, k_att_w)

        fZ_calib[name] = fZ_c
        fZ_wr[name]    = fZ_w
        print(f"  done.", flush=True)

    _plot(gamma_grid, fZ_calib, fZ_wr, nH_total)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _load_andrews(pah_key: str) -> dict:
    """Load digitised Andrews 2016 charge fractions for a given PAH (all derivatives).

    pah_key: 'C24', 'C54', or 'C96'.
    Returns {Z: ndarray(N,2)} with columns [gamma, fraction].
    """
    ext = ROOT / 'external_data'
    data = {}
    for Z, tag in [(-1, 'anion'), (0, 'neutral'), (1, 'cation'), (2, 'dication')]:
        f = ext / f'{pah_key}HN_{tag}_andrews16.csv'
        if f.exists():
            data[Z] = np.loadtxt(f, delimiter=',')
    return data


def _plot(gamma_grid: np.ndarray, fZ_calib: dict, fZ_wr: dict, nH_ref: float) -> None:
    Z_colors = {-1: '#9467bd', 0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    Z_labels = {-1: r'$Z=-1$',  0: r'$Z=0$',  1: r'$Z=+1$',  2: r'$Z=+2$'}
    Z_ls     = {-1: '--',       0: '-',        1: '-',         2: '-.'}

    andrews = {name: _load_andrews(name) for name in _PAH_ORDER}

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 5),
        sharey=True, constrained_layout=True,
    )

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

        for Z, arr in andrews[name].items():
            lbl = f'Andrews+16 $Z={Z:+d}$' if col == 0 else None
            ax.scatter(arr[:, 0], arr[:, 1],
                       color=Z_colors[Z], marker='o', s=30, zorder=5,
                       edgecolors='k', linewidths=0.4, label=lbl)

        ax.set_xscale('log')
        ax.set_xlim(gamma_grid[0], gamma_grid[-1])
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r'$\gamma = G_0\,\sqrt{T}\,/\,n_e$', fontsize=11)
        ax.set_title(pdef['label'], fontsize=11)
        if col == 0:
            ax.set_ylabel('Charge-state fraction', fontsize=11)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)

    fig.suptitle(
        rf'PAH charge distribution (full $N_H$ network, $n_H={nH_ref:.0f}$ cm$^{{-3}}$, '
        r'$T=500\,\mathrm{K}$, Kurucz 15 kK)'
        '\n'
        r'Thick: WR $s_e$ (Cagliari $\alpha$, Li\&Draine $k_r$, $T=500\,\mathrm{K}$)'
        r'   Faded: calibrated $s_e$ (Andrews+16)'
        '\nCircles: Andrews et al. (2016) Fig. 8 [digitised]',
        fontsize=10,
    )

    out = ROOT / 'pah_charge_vs_gamma_full.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
