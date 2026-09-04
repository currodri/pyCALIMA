"""
Test: 5 PAH + 10 graphite + 10 silicate bin discretization vs MW optical properties.

Workflow
--------
  1. For each bin, fit dn/da to the Zubko (2004) BARE-GR-S parametric distribution:
       - Non-last bins: power law dn/da = C * a^{-alpha}, alpha from log-log regression.
       - Last bin per composition: pure exponential dn/da = C * exp(-a/a_c), a_c from
         log-linear regression of log(dn/da) vs a (fitted to full bin range).
     Mass fraction D_i from exact numerical integral of Zubko dn/da in each bin.
     Normalization C back-derived from D_i and the distribution's sintegral.

  2. Export per-bin cross-section tables.

  3. Plot extinction curve vs Zubko (2004) reference.

  4. Plot per-composition distribution panels.

Outputs: model_data/optical_properties_4C6Si/ and results/.

Usage
-----
    python models/dust_radiation/test_4C6Si_mw_comparison.py [--skip-export] [--tag TAG]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

from pycalima.models.grain_size_config import set_config_path, get_lognormal_parameters, get_model_data_dir
from pycalima.models.grain_distributions import PowerLaw_Distribution, Exponential_Distribution

CONFIG_PATH = str(_REPO_ROOT / 'models' / 'grain_size_distribution_4C6Si.json')
set_config_path(CONFIG_PATH)
OUTPUT_DIR  = str(get_model_data_dir() / 'optical_properties')
RESULTS_DIR = str(_REPO_ROOT / 'results')

GRAPHITE_BINS = [f'DustBin_{i:02d}' for i in range(1, 11)]   # DustBin_01 … DustBin_10
SILICATE_BINS = [f'DustBin_{i:02d}' for i in range(11, 21)]  # DustBin_11 … DustBin_20
DUST_BINS     = GRAPHITE_BINS + SILICATE_BINS
PAH_BINS      = ['PAHBin_01', 'PAHBin_02', 'PAHBin_03', 'PAHBin_04', 'PAHBin_05']

# Last bin of each composition → Exponential_Distribution
# All other bins               → PowerLaw_Distribution
def _build_dist_class_map():
    m = {}
    for bins in (PAH_BINS, GRAPHITE_BINS, SILICATE_BINS):
        m[bins[-1]] = Exponential_Distribution
        for b in bins[:-1]:
            m[b] = PowerLaw_Distribution
    return m

DIST_CLASS_MAP = _build_dist_class_map()


def fit_zubko_powlaw_distributions() -> dict:
    """Fit piecewise distributions to Zubko (2004) BARE-GR-S.

    Non-last bins: power law, alpha from log-log regression over [amin, amax].
    Last bins:     exponential, a_c (µm) from log-linear regression over [amin, amax].

    Mass fraction D_i always from exact Zubko integral; C back-derived from sintegral.
    The 'alpha' key in the return dict holds alpha for power-law bins and a_c (µm)
    for exponential bins.
    """
    from pycalima.models.dust_radiation.dust_oppacity import (
        _zubko_dnda_graphite, _zubko_dnda_silicate, _zubko_dnda_pah,
    )

    set_config_path(CONFIG_PATH)

    zubko_func = {}
    for b in PAH_BINS:      zubko_func[b] = _zubko_dnda_pah
    for b in GRAPHITE_BINS: zubko_func[b] = _zubko_dnda_graphite
    for b in SILICATE_BINS: zubko_func[b] = _zubko_dnda_silicate

    results = {}
    for bin_id in PAH_BINS + DUST_BINS:
        p = get_lognormal_parameters(bin_id)
        amin, amax, rho = p['amin'], p['amax'], p['s']

        a    = np.logspace(np.log10(amin), np.log10(amax), 500)
        dnda = zubko_func[bin_id](a)

        _cls = DIST_CLASS_MAP.get(bin_id, PowerLaw_Distribution)

        if _cls is Exponential_Distribution:
            # log(dn/da) = log(C) - a/a_c  →  linear fit vs a (not log a)
            valid = dnda > 0
            A = np.column_stack([np.ones(valid.sum()), a[valid]])
            b_vec = np.log(dnda[valid])
            coeff, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            a_c   = -1.0 / coeff[1]   # µm
            alpha = a_c                # stored in alpha slot; units: µm
            # sintegral in µm units: ∫ a^3 * exp(-a/a_c) da  from amin to 20*amax
            a_g   = np.logspace(np.log10(amin), np.log10(amax * 20.), 800)
            shape = np.exp(-a_g / a_c)
        else:
            # Power law: alpha from log-log regression
            valid  = dnda > 0
            coeffs = np.polyfit(np.log10(a[valid]), np.log10(dnda[valid]), 1)
            alpha  = -coeffs[0]
            a_g    = np.logspace(np.log10(amin), np.log10(amax), 600)
            shape  = a_g**(-alpha)

        # D_i: actual Zubko mass in bin (g / H)
        a_cm      = a * 1e-4
        m_a       = (4. / 3.) * np.pi * rho * a_cm**3
        D_i_exact = np.trapezoid(m_a * dnda, a)

        sintegral = (4. / 3.) * np.pi * rho * (1e-4)**3 * np.trapezoid(
            a_g**3 * shape, a_g
        )
        C = D_i_exact / sintegral

        results[bin_id] = dict(alpha=alpha, C=C, D=D_i_exact,
                               amin=amin, amax=amax, rho=rho)

    D_total = sum(r['D'] for r in results.values())
    for r in results.values():
        r['mass_fraction'] = r['D'] / D_total

    return dict(
        alpha={b: results[b]['alpha'] for b in PAH_BINS + DUST_BINS},
        C    ={b: results[b]['C']     for b in PAH_BINS + DUST_BINS},
        D    ={b: results[b]['D']     for b in PAH_BINS + DUST_BINS},
        mass_fraction={b: results[b]['mass_fraction'] for b in PAH_BINS + DUST_BINS},
        dust_mass_fractions=[results[b]['mass_fraction'] for b in DUST_BINS],
        pah_mass_fractions =[results[b]['mass_fraction'] for b in PAH_BINS],
        D_total=D_total,
    )


def update_json_sigma(fit_results: dict) -> None:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    bin_order = [b['id'] for b in cfg['bins']]
    new_sigma = []
    for bin_id in bin_order:
        v = fit_results['alpha'].get(bin_id)
        if v is not None:
            new_sigma.append(round(float(v), 6))
        else:
            idx = bin_order.index(bin_id)
            new_sigma.append(cfg['basic']['sigma'][idx])
    cfg['basic']['sigma'] = new_sigma
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  Updated sigma in {CONFIG_PATH}")


def print_fit_summary(fit_results: dict) -> None:
    D_total = fit_results['D_total']
    D_pah   = sum(fit_results['D'][b] for b in PAH_BINS)
    D_gra   = sum(fit_results['D'][b] for b in GRAPHITE_BINS)
    D_sil   = sum(fit_results['D'][b] for b in SILICATE_BINS)

    print(f"\n{'Bin':<14} {'amin':>7} {'amax':>7} {'param':>14}  {'mf':>10}  type")
    print("-" * 70)
    for bin_id in PAH_BINS + DUST_BINS:
        p    = get_lognormal_parameters(bin_id)
        mf   = fit_results['mass_fraction'][bin_id]
        v    = fit_results['alpha'][bin_id]
        _cls = DIST_CLASS_MAP.get(bin_id, PowerLaw_Distribution)
        label = f'a_c={v:.4f}µm' if _cls is Exponential_Distribution else f'α={v:.3f}'
        print(f"  {bin_id:<12} {p['amin']:7.4f} {p['amax']:7.4f} {label:>14}  {mf:10.4f}")

    print(f"\n  Total D = {D_total:.4e} g/H")
    print(f"  PAH      {D_pah/D_total:.4f}  (Zubko ref 0.0457)")
    print(f"  Graphite {D_gra/D_total:.4f}  (Zubko ref 0.2947)")
    print(f"  Silicate {D_sil/D_total:.4f}  (Zubko ref 0.6596)")


def stage1_fit() -> dict:
    print("\n=== Stage 1: Fitting distributions to Zubko (2004) ===")
    fit_results = fit_zubko_powlaw_distributions()
    print_fit_summary(fit_results)
    update_json_sigma(fit_results)
    return fit_results


def stage2_export() -> None:
    from pycalima.models.dust_radiation.export_dust_optical_properties import export_dust_optical_properties
    from pycalima.models.PAH_radiation.pah_oppacity import export_pah_optical_properties

    set_config_path(CONFIG_PATH)
    print("\n=== Stage 2: Exporting per-bin optical properties ===")
    print(f"  Config  : {CONFIG_PATH}")
    print(f"  Out dir : {OUTPUT_DIR}")

    export_dust_optical_properties(
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        cabs_method='mie',
        distribution_class_map=DIST_CLASS_MAP,
    )
    export_pah_optical_properties(
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
    )


def stage3_compare(fit_results: dict, out_png: str | None = None) -> dict:
    from pycalima.models.dust_radiation.dust_oppacity import plot_extinction_from_massfractions

    set_config_path(CONFIG_PATH)
    if out_png is None:
        out_png = str(Path(RESULTS_DIR) / 'test_4C6Si_powlaw_comparison.png')
    print("\n=== Stage 3: Extinction comparison ===")
    print(f"  Saving plot to {out_png}")

    return plot_extinction_from_massfractions(
        dust_bins=DUST_BINS,
        dust_mass_fractions=fit_results['dust_mass_fractions'],
        pah_bins=PAH_BINS,
        pah_mass_fractions=fit_results['pah_mass_fractions'],
        optical_dir=OUTPUT_DIR,
        out_png=out_png,
        pah_state='neutral',
        verbose=True,
        distribution_class_map=DIST_CLASS_MAP,
        mdust_per_H=fit_results['D_total'],
    )


def stage4_distribution_panels(fit_results: dict, out_png: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pycalima.models.dust_radiation.dust_oppacity import (
        _zubko_dnda_graphite, _zubko_dnda_silicate, _zubko_dnda_pah,
    )

    set_config_path(CONFIG_PATH)
    if out_png is None:
        out_png = str(Path(RESULTS_DIR) / 'test_4C6Si_distributions.png')
    print(f"\n=== Stage 4: Per-composition distribution panels ===")
    print(f"  Saving plot to {out_png}")

    sns.set_theme(style='white')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=200, sharey=False)

    specs = [
        ('PAH',      PAH_BINS,      _zubko_dnda_pah,      0.00035, 0.005,  axes[0]),
        ('Graphite', GRAPHITE_BINS, _zubko_dnda_graphite, 0.00035, 0.330,  axes[1]),
        ('Silicate', SILICATE_BINS, _zubko_dnda_silicate, 0.00035, 0.370,  axes[2]),
    ]

    for comp_name, bin_ids, zubko_func, zamin, zamax, ax in specs:
        palette = sns.color_palette('tab10', n_colors=len(bin_ids))

        az    = np.logspace(np.log10(zamin), np.log10(zamax), 400)
        az_cm = az * 1e-4
        y_z   = az_cm**4 * zubko_func(az) * 1e4
        ax.plot(az, y_z, 'k-', lw=2, label='Zubko (2004)', zorder=5)

        all_y_pos = []
        for i, bin_id in enumerate(bin_ids):
            param = fit_results['alpha'][bin_id]
            C     = fit_results['C'][bin_id]
            p     = get_lognormal_parameters(bin_id)
            amin, amax = p['amin'], p['amax']

            _cls = DIST_CLASS_MAP.get(bin_id, PowerLaw_Distribution)
            if _cls is Exponential_Distribution:
                a_c  = param   # µm
                a    = np.logspace(np.log10(amin), np.log10(amax * 5.), 300)
                dnda = C * np.exp(-a / a_c)
                lbl  = fr'{bin_id}  $a_c$={a_c:.3f}µm'
            else:
                alpha = param
                a    = np.logspace(np.log10(amin), np.log10(amax), 200)
                dnda = C * a**(-alpha)
                lbl  = fr'{bin_id}  $\alpha$={alpha:.2f}'

            a_cm  = a * 1e-4
            y_fit = a_cm**4 * dnda * 1e4
            all_y_pos.extend(y_fit[y_fit > 0].tolist())
            ax.plot(a, y_fit, color=palette[i], lw=1.8, label=lbl)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$a\;[\mu\mathrm{m}]$', fontsize=12)
        ax.set_ylabel(r'$a^4\,n(a)\;[\mathrm{cm}^3\,\mathrm{H}^{-1}]$', fontsize=11)
        ax.set_title(comp_name, fontsize=13)
        if all_y_pos:
            y_peak = max(all_y_pos)
            ax.set_ylim(bottom=y_peak * 1e-5, top=y_peak * 3.)
        ax.legend(fontsize=6.5, frameon=False, loc='best')
        ax.grid(alpha=0.2, which='both')

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main() -> None:
    import datetime
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    ts     = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f"_{args.tag}" if args.tag else f"_{ts}"

    fit_results = stage1_fit()

    if not args.skip_export:
        stage2_export()

    out_png  = str(Path(RESULTS_DIR) / f'test_4C6Si_powlaw{suffix}.png')
    out_dist = str(Path(RESULTS_DIR) / f'test_4C6Si_distributions{suffix}.png')

    stage3_compare(fit_results, out_png=out_png)
    stage4_distribution_panels(fit_results, out_png=out_dist)

    print(f"\nDone. Plots saved to:")
    print(f"  {out_png}")
    print(f"  {out_dist}")


if __name__ == '__main__':
    main()
