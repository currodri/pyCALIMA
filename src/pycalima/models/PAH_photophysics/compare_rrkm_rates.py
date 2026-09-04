"""
compare_rrkm_rates.py
=====================
Compare PAH H-loss and H2-loss photodissociation rates from our Method B
(GD89 temperature distribution + RRKM Arrhenius formula, PAHdb modes) against
the 4th-order polynomial fits from Andrews et al. (2016) Tables B.1-B.3 for
coronene C24H12, circumcoronene C54H18, and circumcircumcoronene C96H24.

Workflow
--------
For each PAH size and each G0 in G0_GRID:
  1. Run _worker (reproduce_andrews16_fig9) → one Method B rate per RRKM class.
  2. For every (Z, Nh) state in the Andrews table, map the state to its RRKM
     class (H_even_duo / H_even_solo / H_odd / superH_neutral / superH_cation).
  3. Evaluate the Andrews polynomial k(G0) and compare to the class rate.

Output: three figures (rrkm_vs_andrews16_{C24,C54,C96}.png) and a console
discrepancy table with mean and RMS of log10(RRKM / Andrews) per state.

Why discrepancies exist
-----------------------
Andrews and our code use the *identical* RRKM Arrhenius formula from Tielens
(2005), but three inputs differ:

  1. E-T relation (dominant, enters exponentially):
     We solve U_QHO(T_m) = E using PAHdb vibrational modes.
     Andrews use B3LYP/6-31G* DFT modes for each PAH.
     A 5-10 % difference in mode frequencies shifts T_m by ~100-300 K, which
     changes exp(−E_act / k_B T_e) by one to three orders of magnitude.

  2. IR cooling rate k_IR:
     We compute it from PAHdb Einstein A coefficients; Andrews from DFT modes.
     Ratio k_IR_PAHdb / k_IR_Andrews ≈ 0.53-0.87 for C54H18 (see session notes).

  3. UV absorption cross-section:
     We use Li & Draine (2001) grain tables at radius afromNc(Nc).
     Andrews use molecule-specific quantum-chemistry cross-sections.
     This scales the total rate linearly (not exponentially).

Summary of discrepancy (from a representative run):
  C24 H_even_duo (Z=0, Nh=12) : mean −1.89 dex   (RRKM 77× too small)
  C54 H_even_duo (Z=0, Nh=18) : mean +0.11 dex   (RRKM ≈ 1.3× Andrews — fortuitous)
  C96 H_even_duo (Z=0, Nh=24) : mean +1.01 dex   (RRKM 10× too large)
  super-H states (all sizes)   : −2.8 to −9.6 dex (additional error from using
                                  normal-H modes for the sp3-H molecule)
  Global median RMS             : 2.8 dex

The "factor-of-2" agreement seen in diagnose_temperature_distribution.py holds
only for C54H18 neutral ground state, which is the case that diagnostic uses.

Usage
-----
    python -m models.PAH_photophysics.compare_rrkm_rates
"""

from __future__ import annotations

import warnings
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from multiprocessing import Pool, cpu_count

from pycalima.models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_u_E
from pycalima.models.PAH_photophysics.pah_h_state import compute_solo_duo_counts
from pycalima.models.PAH_photophysics.reproduce_andrews16_fig9 import (
    _STATES_DIR, _HV_EV, _worker,
)
from pycalima import _paths

_EXT = _paths.get_external_data_path()

# ── PAH configuration ─────────────────────────────────────────────────────────
PAH_CFG = {
    'C24': dict(
        Nc=24,  Nh0=12, solo=0,  duo=12,
        label=r'Coronene $C_{24}H_{12}$',
        table='table_B1_coronene.csv',
        modes='C24H12_0.dat',
        IP1=7.20, IP2=11.50,
    ),
    'C54': dict(
        Nc=54,  Nh0=18, solo=6,  duo=12,
        label=r'Circumcoronene $C_{54}H_{18}$',
        table='table_B2_circumcoronene.csv',
        modes='C54H18_0.dat',
        IP1=6.14, IP2=8.91,
    ),
    'C96': dict(
        Nc=96,  Nh0=24, solo=12, duo=12,
        label=r'Circumcircumcoronene $C_{96}H_{24}$',
        table='table_B3_circumcircumcoronene.csv',
        modes='C96H24_0.dat',
        IP1=5.68, IP2=8.24,
    ),
}

# G0 grid for evaluation
NG0 = 40
G0_GRID = np.logspace(0, 5, NG0)   # 1 to 1e5

# ── RRKM class colours and labels ─────────────────────────────────────────────
CLASS_COLOR = {
    'H_even_duo':     '#1f77b4',
    'H_even_solo':    '#aec7e8',
    'H_odd':          '#ff7f0e',
    'superH_neutral': '#2ca02c',
    'superH_cation':  '#d62728',
}
CLASS_LABEL = {
    'H_even_duo':     r'H$_\mathrm{even}$ (duo) — $E_a=4.60$ eV',
    'H_even_solo':    r'H$_\mathrm{even}$ (solo) — $E_a=4.60$ eV',
    'H_odd':          r'H$_\mathrm{odd}$ — $E_a=4.10$ eV',
    'superH_neutral': r'super-H, $Z\leq0$ — $E_a=1.40$ eV',
    'superH_cation':  r'super-H, $Z>0$ — $E_a=1.55$ eV',
}


# ── Helper: polynomial rate ───────────────────────────────────────────────────

def poly_rate(coeffs: np.ndarray, G0: np.ndarray) -> np.ndarray:
    """
    Evaluate Andrews (2016) 4th-order log-polynomial at each G0.

        k(G0) = 10^( p0 + p1 x + p2 x^2 + p3 x^3 + p4 x^4 ),  x = log10(G0)

    coeffs is the array [p0, p1, p2, p3, p4] from the CSV tables.
    """
    x = np.log10(G0)
    log_k = sum(coeffs[i] * x**i for i in range(len(coeffs)))
    return 10.0 ** log_k


def rrkm_class(Nh: int, Nh0: int, Z: int, solo: int, duo: int) -> str:
    """
    Map a (Nh, Z) hydrogenation state to its RRKM dissociation class.

    Mirrors _build_kdis in reproduce_andrews16_fig9.py:
      Nh > Nh0     → super-H (sp3-bonded extra H)
      Nh odd       → H_odd (lone peripheral H, lower E_act = 4.10 eV)
      Nh even, duo → H_even_duo (adjacent pair, H2 loss possible)
      Nh even, solo→ H_even_solo (isolated H, H2 channel suppressed)
    """
    if Nh > Nh0:
        return 'superH_cation' if Z > 0 else 'superH_neutral'
    if Nh % 2 == 1:
        return 'H_odd'
    state = compute_solo_duo_counts(Nh, solo, duo)
    return 'H_even_duo' if state['H2_loss_possible'] else 'H_even_solo'


# ── Load Andrews+16 tables ────────────────────────────────────────────────────

def load_table(fname: str) -> pd.DataFrame:
    df = pd.read_csv(_EXT / fname)
    H_cols  = ['H_loss_p0',  'H_loss_p1',  'H_loss_p2',  'H_loss_p3',  'H_loss_p4']
    H2_cols = ['H2_loss_p0', 'H2_loss_p1', 'H2_loss_p2', 'H2_loss_p3', 'H2_loss_p4']
    df['has_H2'] = df[H2_cols].notna().all(axis=1)
    for col in H_cols + H2_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Computing Kurucz G0 base ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    print(f"  G0_base = {G0_base:.4e}\n", flush=True)

    all_discrepancies: list[dict] = []

    for pah_key, cfg in PAH_CFG.items():
        print(f"{'='*60}", flush=True)
        print(f"  {cfg['label']}", flush=True)
        print(f"{'='*60}", flush=True)

        # ── Build cross-section table ──
        a0 = get_absorption_cross_section.__module__   # dummy import check
        from pycalima.models.PAH_photophysics.pah_charge_utils import afromNc
        a0_cm = afromNc(cfg['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0_cm)
        E_cs = _HV_EV / w
        idx  = np.argsort(E_cs)
        xsect = np.column_stack([E_cs[idx], C_abs[idx]])

        modes_path = str(_STATES_DIR / cfg['modes'])

        # ── Run workers for G0 grid ──
        tasks = [
            (pah_key, modes_path, xsect, float(G0), G0_base, cfg['IP1'], cfg['IP2'])
            for G0 in G0_GRID
        ]
        n_workers = max(1, min(cpu_count() - 1, NG0))
        print(f"  Running {NG0} workers ...", flush=True)
        with Pool(n_workers) as pool:
            raw = pool.map(_worker, tasks)
        # pool.map preserves order
        rrkm_rates = [r[2] for r in raw]   # list of rate dicts, one per G0 point

        # ── Load Andrews+16 polynomial table ──
        df = load_table(cfg['table'])

        # ── Evaluate rates and collect discrepancies ──
        H_cols  = ['H_loss_p0',  'H_loss_p1',  'H_loss_p2',  'H_loss_p3',  'H_loss_p4']
        H2_cols = ['H2_loss_p0', 'H2_loss_p1', 'H2_loss_p2', 'H2_loss_p3', 'H2_loss_p4']

        # Store per-state results
        results = []
        for _, row in df.iterrows():
            Z   = int(row['Z'])
            Nh  = int(row['NH'])
            cls = rrkm_class(Nh, cfg['Nh0'], Z, cfg['solo'], cfg['duo'])

            # Andrews+16 polynomial rates
            h_coeffs = row[H_cols].values.astype(float)
            k_H_poly = poly_rate(h_coeffs, G0_GRID)

            k_H2_poly = None
            if row['has_H2']:
                h2_coeffs = row[H2_cols].values.astype(float)
                k_H2_poly = poly_rate(h2_coeffs, G0_GRID)

            # RRKM model rates (class-based)
            cls_key = cls.replace('H_even_duo', 'H_even_duo').replace('H_even_solo', 'H_even_solo')
            # _worker returns rrkm[cls] = (Y_H, Y_H2)
            k_H_rrkm  = np.array([rrkm_rates[ig][cls][0] for ig in range(NG0)])
            k_H2_rrkm = np.array([rrkm_rates[ig][cls][1] for ig in range(NG0)])

            # Discrepancy: log10(RRKM / Andrews) for H-loss
            mask = (k_H_poly > 0) & (k_H_rrkm > 0)
            if mask.any():
                log_ratio = np.log10(k_H_rrkm[mask] / k_H_poly[mask])
                rms  = float(np.sqrt(np.mean(log_ratio**2)))
                mean = float(np.mean(log_ratio))
            else:
                rms = mean = np.nan

            results.append(dict(
                Z=Z, Nh=Nh, cls=cls,
                k_H_poly=k_H_poly, k_H2_poly=k_H2_poly,
                k_H_rrkm=k_H_rrkm, k_H2_rrkm=k_H2_rrkm,
                log_ratio_mean=mean, log_ratio_rms=rms,
            ))
            all_discrepancies.append(dict(pah=pah_key, Z=Z, Nh=Nh, cls=cls,
                                          mean=mean, rms=rms))

        _plot_pah(pah_key, cfg, results, G0_GRID)
        _print_discrepancy_table(pah_key, results)

    _print_summary(all_discrepancies)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_pah(pah_key: str, cfg: dict, results: list, G0_grid: np.ndarray) -> None:
    """One figure per PAH: 2 rows × 2 cols.

    (0,0) H-loss: Andrews poly (one curve per state, coloured by RRKM class)
                  RRKM per class: bold dashed, same colour
    (0,1) H2-loss: same layout, only states with H2-loss data
    (1,0) H-loss ratio  RRKM / Andrews  per state
    (1,1) H2-loss ratio
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True,
                              sharex=True)

    ax_H, ax_H2 = axes[0]
    ax_Hr, ax_H2r = axes[1]

    # Track which RRKM classes have been plotted (for legend)
    plotted_classes_H  = set()
    plotted_classes_H2 = set()

    # Linestyle cycle within a class to separate states
    ls_cycle = ['-', '--', ':', '-.', (0, (3,1,1,1))]

    class_state_counter: dict[str, int] = {}

    for rec in results:
        Z, Nh, cls = rec['Z'], rec['Nh'], rec['cls']
        color = CLASS_COLOR[cls]
        ci = class_state_counter.get(cls, 0)
        class_state_counter[cls] = ci + 1
        ls = ls_cycle[ci % len(ls_cycle)]
        lw = 1.0
        state_lbl = f'$Z={Z:+d}$, $N_H={Nh}$'

        # ── H-loss rate plot ──
        ax_H.loglog(G0_grid, rec['k_H_poly'], color=color, lw=lw,
                    ls=ls, alpha=0.75, label=state_lbl)

        if cls not in plotted_classes_H:
            ax_H.loglog(G0_grid, rec['k_H_rrkm'], color=color,
                        lw=2.5, ls='--', alpha=1.0,
                        label=CLASS_LABEL[cls])
            plotted_classes_H.add(cls)

        # H-loss ratio
        mask = (rec['k_H_poly'] > 0) & (rec['k_H_rrkm'] > 0)
        if mask.any():
            ratio = rec['k_H_rrkm'][mask] / rec['k_H_poly'][mask]
            ax_Hr.semilogx(G0_grid[mask], np.log10(ratio),
                           color=color, lw=lw, ls=ls, alpha=0.75)

        # ── H2-loss rate plot ──
        if rec['k_H2_poly'] is not None:
            ax_H2.loglog(G0_grid, rec['k_H2_poly'], color=color, lw=lw,
                         ls=ls, alpha=0.75, label=state_lbl)
            if cls not in plotted_classes_H2:
                ax_H2.loglog(G0_grid, rec['k_H2_rrkm'], color=color,
                             lw=2.5, ls='--', alpha=1.0,
                             label=CLASS_LABEL[cls])
                plotted_classes_H2.add(cls)

            mask2 = (rec['k_H2_poly'] > 0) & (rec['k_H2_rrkm'] > 0)
            if mask2.any():
                ratio2 = rec['k_H2_rrkm'][mask2] / rec['k_H2_poly'][mask2]
                ax_H2r.semilogx(G0_grid[mask2], np.log10(ratio2),
                                color=color, lw=lw, ls=ls, alpha=0.75)

    # ── Decorations ──
    for ax in (ax_H, ax_H2):
        ax.set_ylabel(r'$k$ [s$^{-1}$]', fontsize=11)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
        ax.set_xlim(G0_grid[0], G0_grid[-1])

    for ax in (ax_Hr, ax_H2r):
        ax.axhline(0, color='k', lw=0.8, ls='-')
        ax.axhline( 0.5, color='k', lw=0.5, ls=':')
        ax.axhline(-0.5, color='k', lw=0.5, ls=':')
        ax.set_ylabel(r'$\log_{10}(k_\mathrm{RRKM}/k_\mathrm{A16})$', fontsize=11)
        ax.set_xlabel(r'$G_0$', fontsize=11)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
        ax.set_xlim(G0_grid[0], G0_grid[-1])

    ax_H.set_title('H-loss rate', fontsize=11)
    ax_H2.set_title('H$_2$-loss rate', fontsize=11)
    ax_Hr.set_title('H-loss discrepancy  (0 = perfect agreement)', fontsize=10)
    ax_H2r.set_title('H$_2$-loss discrepancy', fontsize=10)

    # Legend: separate handles for Andrews poly and RRKM model
    proxy_solid = mlines.Line2D([], [], color='grey', lw=1.0, ls='-',
                                alpha=0.75, label='Andrews+16 polynomial (per state)')
    proxy_dash  = mlines.Line2D([], [], color='grey', lw=2.5, ls='--',
                                label='RRKM model (per class)')
    ax_H.legend(
        [proxy_solid, proxy_dash]
        + [mlines.Line2D([], [], color=CLASS_COLOR[c], lw=2.0,
                         label=CLASS_LABEL[c]) for c in CLASS_COLOR if c in plotted_classes_H],
        ['Andrews+16 poly', 'RRKM model']
        + [CLASS_LABEL[c] for c in CLASS_COLOR if c in plotted_classes_H],
        fontsize=7, ncol=1, loc='upper left',
    )
    if plotted_classes_H2:
        ax_H2.legend(fontsize=7, loc='upper left')

    fig.suptitle(
        cfg['label'] + r'  —  RRKM vs Andrews+16 polynomial fits'
        '\n'
        r'Solid thin: Andrews+16 per state  |  Dashed bold: our RRKM (per class)',
        fontsize=10,
    )

    out = _paths.get_plots_dir('pah_photophysics') / f'rrkm_vs_andrews16_{pah_key}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Figure saved → {out}", flush=True)
    plt.close(fig)


# ── Console tables ────────────────────────────────────────────────────────────

def _print_discrepancy_table(pah_key: str, results: list) -> None:
    print(f"\n  Discrepancy table (H-loss): log10(RRKM / Andrews+16)")
    print(f"  {'Z':>4}  {'Nh':>4}  {'class':>20}  {'mean':>8}  {'RMS':>8}")
    print("  " + "-"*50)
    for rec in sorted(results, key=lambda r: (r['cls'], r['Z'], r['Nh'])):
        print(f"  {rec['Z']:>4d}  {rec['Nh']:>4d}  {rec['cls']:>20s}  "
              f"{rec['log_ratio_mean']:>+8.3f}  {rec['log_ratio_rms']:>8.3f}")
    print()


def _print_summary(all_disc: list[dict]) -> None:
    print("\n" + "="*65)
    print("  GLOBAL SUMMARY — median |log10 ratio| per RRKM class")
    print("="*65)
    import collections
    by_class: dict[str, list] = collections.defaultdict(list)
    for d in all_disc:
        if np.isfinite(d['rms']):
            by_class[d['cls']].append(d['rms'])

    print(f"  {'RRKM class':>22s}  {'N states':>8}  {'median RMS':>12}  {'max RMS':>10}")
    print("  " + "-"*55)
    for cls in ['H_even_duo', 'H_even_solo', 'H_odd', 'superH_neutral', 'superH_cation']:
        vals = by_class.get(cls, [])
        if vals:
            print(f"  {cls:>22s}  {len(vals):>8d}  {np.median(vals):>12.3f}  {np.max(vals):>10.3f}")

    print()
    all_rms = [d['rms'] for d in all_disc if np.isfinite(d['rms'])]
    print(f"  Overall: N={len(all_rms)}  median RMS={np.median(all_rms):.3f}"
          f"  max RMS={np.max(all_rms):.3f}  "
          f"mean bias={np.mean([d['mean'] for d in all_disc if np.isfinite(d['mean'])]):.3f}")
    print()
    print("  Note: all values in dex (log10 units).")
    print("  RMS < 0.30 → better than factor-of-2 accuracy.")
    print("  RMS < 1.00 → order-of-magnitude accuracy.")
    print("  Positive mean bias = RRKM overestimates Andrews+16.")
    print("  A large mean bias with large RMS usually indicates a systematic")
    print("  mode-set mismatch (see module docstring for explanation).")


if __name__ == '__main__':
    main()
