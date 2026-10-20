#!/usr/bin/env python3
"""Dust sublimation around a quasar: compare pyCALIMA with the literature.

pyCALIMA computes, for one grain of radius a and composition X at distance r
from a point-source AGN of bolometric luminosity L:

* T_eq(a, X, r): radiative equilibrium with the Draine & Lee (1984) Q_abs
  (``dust_sublimation.compute_bin_temperature``, bracket up to 4000 K);
* T_sub(a, X, t): the temperature at which the GD89 lifetime a/|da/dt|
  equals an exposure time t (``dust_sublimation.sublimation_timescale``);
* R_sub(a, X, t): the radius where T_eq = T_sub.

The SED is the SKIRTOR one (Stalevski et al. 2012, 2016; Schartmann et al.
2005): lambda L_lambda ~ lambda^1.2 (1-10 nm), flat (10-100 nm),
lambda^-0.5 (0.1-5 um), lambda^-3 beyond, emitted isotropically. L_bol is
the 1 nm - 1 mm integral.

Each literature relation is evaluated with its own assumptions (grain size,
composition, T_sub, luminosity definition) and set against pyCALIMA's radius
for the same grain at the same temperature. Equations as printed in:

  Barvainis 1987, ApJ 320, 537, eqs. 4-5 (graphite, a = 0.05 um, L_uv)
  Kishimoto et al. 2007, A&A 476, 713, eqs. 1 and 3 (L_UV = 6 nu L_nu(V))
  Nenkova et al. 2008b, ApJ 685, 160, eq. 1 (ISM mix, L_bol)
  Mor & Netzer 2012, MNRAS 420, 526, eqs. 1-2 (L_bol)
  Baskin & Laor 2018, MNRAS 474, 1970, eqs. 12, 32, 33 (L_bol)
  Waxman & Draine 2000, ApJ 537, 796, eq. 10 with the GD89 constants
  GRAVITY Collaboration 2024, A&A 690, A76, eq. 2 and Table 4 (L_bol)

pyCALIMA uses the GD89 dataset here by default, since the literature
relations assume GD89-like kinetics; ``--dataset fiducial`` uses the
laboratory sublimation data (``sublimation_kinetics``).

Usage
-----
    python diagnostics/dust_radiation/check_sublimation_radius_agn.py [--dataset fiducial]
"""
import argparse

import numpy as np
from scipy.optimize import brentq

from pycalima import _paths
from pycalima.models.dust_radiation import dust_sublimation as ds
from pycalima.models.dust_radiation import sublimation_kinetics as sk
from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections

PC = 3.0856775814913673e18
YR = 3.1556952e7
DAY = 86400.0
L_BOL = 1.0e46  # [erg s^-1]
EXPOSURES = {'1 day': DAY, '1 yr': YR, '1 kyr': 1e3 * YR}
COLORS = {'graphite': '#2a78d6', 'silicate': '#eb6834'}

LAM_UM = np.logspace(-3, 3, 3000)
WAV_CM = LAM_UM * 1e-4


def skirtor_shape(lam_um):
    """SKIRTOR lambda L_lambda shape, continuous, arbitrary normalisation."""
    l = np.asarray(lam_um, dtype=float)
    return np.where(l < 0.01, (l / 0.01) ** 1.2,
           np.where(l < 0.1, 1.0,
           np.where(l < 5.0, (l / 0.1) ** -0.5, 50.0 ** -0.5 * (l / 5.0) ** -3)))


_SHAPE = skirtor_shape(LAM_UM)
_NORM = L_BOL / np.trapezoid(_SHAPE / LAM_UM, LAM_UM)
L_LAM_CM = _NORM * _SHAPE / LAM_UM * 1e4                 # [erg s^-1 cm^-1]
NU_L_NU_V = _NORM * float(skirtor_shape(0.55))         # lambda L_lambda at V
L_UV = 6.0 * NU_L_NU_V                                 # Kishimoto+07 convention


def flux_at(r_pc):
    """F_lambda = L_lambda / 4 pi r^2, i.e. the 4 pi J_lambda the solver wants."""
    return L_LAM_CM / (4.0 * np.pi * (r_pc * PC) ** 2)


_XS_CACHE = {}


def _cross_section(material, a_um):
    key = (material, a_um)
    if key not in _XS_CACHE:
        _, wav, _, C_abs, _ = interpolate_cross_sections(material, a_um)
        _XS_CACHE[key] = (wav, C_abs)
    return _XS_CACHE[key]


def t_eq(material, a_um, r_pc):
    wav, C_abs = _cross_section(material, a_um)
    return ds.compute_bin_temperature(a_um * 1e-4, wav, C_abs, WAV_CM, flux_at(r_pc))


def radius_at_temperature(material, a_um, T):
    """Radius [pc] where T_eq = T, for L_bol = 1e46."""
    return 10 ** brentq(lambda lr: t_eq(material, a_um, 10 ** lr) - T, -4.0, 3.0,
                        xtol=1e-5)


def t_sub(material, a_um, t_exp, timescale=ds.sublimation_timescale):
    return brentq(lambda T: np.log(timescale(material, a_um * 1e-4, T) / t_exp),
                  500.0, 6000.0, xtol=1e-3)


def literature_rows():
    """(label, material, a_um, T_sub, R_lit [pc]) at L_bol = 1e46."""
    L_uv46 = L_UV / 1e46
    L46 = L_BOL / 1e46
    return [
        ('Barvainis 87 eq.5 (L_uv = 6 nuLnu(V))', 'graphite', 0.05, 1500.0,
         1.3 * L_uv46 ** 0.5),
        ('Kishimoto+07 eq.1, a = 0.1 um', 'graphite', 0.1, 1500.0,
         1.3 * L_uv46 ** 0.5 * (0.1 / 0.05) ** -0.5),
        ('Kishimoto+07 eq.1, a = 0.3 um', 'graphite', 0.3, 1500.0,
         1.3 * L_uv46 ** 0.5 * (0.3 / 0.05) ** -0.5),
        ('Mor & Netzer 12 eq.2 (graphite, a = 0.05)', 'graphite', 0.05, 1800.0,
         0.5 * L46 ** 0.5),
        ('Mor & Netzer 12 eq.2 (graphite, a = 0.1)', 'graphite', 0.1, 1800.0,
         0.5 * L46 ** 0.5),
        ('Mor & Netzer 12 eq.1 (silicate, a = 0.1)', 'silicate', 0.1, 1500.0,
         1.3 * L46 ** 0.5),
        ('Nenkova+08b eq.1 vs silicate a = 0.1', 'silicate', 0.1, 1500.0,
         0.4 * (L_BOL / 1e45) ** 0.5),
        ('Baskin & Laor 18 eq.12 (blackbody) vs 10 um', 'graphite', 10.0, 2000.0,
         0.15 * L46 ** 0.5),
    ]


def observed_radii():
    """Hot-dust radii [pc] measured at L_bol = 1e46."""
    lg = np.log10(L_BOL) - 45.1
    return {
        'K-band reverberation (GRAVITY 24)': 10 ** (-0.95 + 0.44 * lg),
        'K-band interferometry (GRAVITY 24)': 10 ** (-0.55 + 0.45 * lg),
        'K-band reverberation (Kishimoto+07 eq.3)': 0.47 * (L_UV / 1e46) ** 0.5,
    }


def baskin_laor_tsub(material, n_cm3):
    """Gas-density-dependent T_sub of Baskin & Laor (2018) eqs. 32-33."""
    if material == 'graphite':
        return 81200.0 / (66.003 - np.log(n_cm3))
    return 68100.0 / (67.957 - np.log(n_cm3))


def print_comparison():
    print(f"SKIRTOR SED, L_bol = {L_BOL:.0e} erg/s: 6 nuLnu(V) = {L_UV:.3e} erg/s "
          f"(L_UV/L_bol = {L_UV / L_BOL:.2f})")
    print("\n1. Kinetics: T_sub [K] at which a/|da/dt| = t, pyCALIMA (GD89) vs "
          "WD00 eq. 10 with GD89 constants")
    print(f"{'material':9s} {'a [um]':>7s} " + " ".join(
        f"{k:>14s}" for k in EXPOSURES))
    for material in ('graphite', 'silicate'):
        for a_um in (0.01, 0.1, 1.0, 10.0):
            cells = []
            for t in EXPOSURES.values():
                cells.append(f"{t_sub(material, a_um, t):6.0f}/"
                             f"{t_sub(material, a_um, t, ds.wd00_sublimation_timescale):<6.0f}")
            print(f"{material:9s} {a_um:7.2f} " + " ".join(f"{c:>14s}" for c in cells))

    print("\n2. Temperature profile: Barvainis 87 eq. 4 has r ~ T^-2.8 for "
          "0.05 um graphite")
    r1, r2 = radius_at_temperature('graphite', 0.05, 1400.0), \
        radius_at_temperature('graphite', 0.05, 1600.0)
    slope = np.log(r2 / r1) / np.log(1600.0 / 1400.0)
    print(f"   pyCALIMA d ln r / d ln T at 1400-1600 K: {slope:.2f}")

    print("\n3. Sublimation radius at the paper's own T_sub and grain "
          "(L_bol = 1e46 erg/s)")
    print(f"{'relation':46s} {'T_sub':>6s} {'R_lit':>7s} {'R_CALIMA':>9s} {'ratio':>6s}")
    for label, material, a_um, T, R_lit in literature_rows():
        R = radius_at_temperature(material, a_um, T)
        print(f"{label:46s} {T:6.0f} {R_lit:7.3f} {R:9.3f} {R / R_lit:6.2f}")

    print("\n4. Self-consistent pyCALIMA R_sub [pc] (T_eq = T_sub(a, t)) vs "
          "observed hot-dust radii")
    print(f"{'material':9s} {'a [um]':>7s} " + " ".join(f"{k:>8s}" for k in EXPOSURES))
    for material in ('graphite', 'silicate'):
        for a_um in (0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0):
            R = [radius_at_temperature(material, a_um, t_sub(material, a_um, t))
                 for t in EXPOSURES.values()]
            print(f"{material:9s} {a_um:7.2f} " + " ".join(f"{x:8.3f}" for x in R))
    for label, R in observed_radii().items():
        print(f"   {label:42s} {R:.3f} pc")

    print("\n5. Gas-density dependence (vacuum rates here; with dust_condensation "
          "see check_recondensation_baskin_laor.py). Baskin & Laor 18 eqs. 32-33:")
    for n in (1e5, 1e8, 1e11):
        print(f"   n = {n:.0e} cm^-3: T_sub(graphite) = "
              f"{baskin_laor_tsub('graphite', n):.0f} K, "
              f"T_sub(silicate) = {baskin_laor_tsub('silicate', n):.0f} K")


def plot(out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    a_grid = np.geomspace(0.01, 10.0, 25)
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for material, color in COLORS.items():
        T_1yr = np.array([t_sub(material, a, YR) for a in a_grid])
        T_wd = np.array([t_sub(material, a, YR, ds.wd00_sublimation_timescale)
                         for a in a_grid])
        ax_t.plot(a_grid, T_1yr, color=color, lw=2, label=f'{material}, pyCALIMA')
        ax_t.plot(a_grid, T_wd, color=color, lw=1.5, ls='--',
                  label=f'{material}, WD00 eq. 10')
        ax_t.axhline(baskin_laor_tsub(material, 1e11), color=color, lw=1, ls=':')

        R = {k: np.array([radius_at_temperature(material, a, t_sub(material, a, t))
                          for a in a_grid]) for k, t in EXPOSURES.items()}
        ax_r.fill_between(a_grid, R['1 day'], R['1 kyr'], color=color, alpha=0.2,
                          lw=0)
        ax_r.plot(a_grid, R['1 yr'], color=color, lw=2, label=material)

    ax_t.text(0.012, baskin_laor_tsub('graphite', 1e11) + 15,
              r'Baskin & Laor 18, $n=10^{11}$ cm$^{-3}$', fontsize=8, color='#52514e')
    for label, material, a_um, T, R_lit in literature_rows():
        ax_r.plot(a_um, R_lit, marker='o', ms=6, mfc='white',
                  mec=COLORS[material], mew=1.5, ls='none')
    for label, R in observed_radii().items():
        if 'GRAVITY' in label:
            ax_r.axhline(R, color='#52514e', lw=1, ls='--')
            ax_r.text(0.011, R * 1.06, label, fontsize=8, color='#52514e')

    ax_t.set(xscale='log', xlabel=r'grain radius $a$ [$\mu$m]',
             ylabel=r'$T_{\rm sub}$ for a 1 yr lifetime [K]')
    ax_r.set(xscale='log', yscale='log', xlabel=r'grain radius $a$ [$\mu$m]',
             ylabel=r'$R_{\rm sub}$ at $L_{\rm bol}=10^{46}$ erg s$^{-1}$ [pc]')
    ax_t.legend(frameon=False, fontsize=8)
    ax_r.legend(frameon=False, fontsize=8, loc='upper right',
                title='lines: 1 yr lifetime, band: 1 day - 1 kyr\ncircles: literature, each at its own $T_{\\rm sub}$',
                title_fontsize=8)
    for ax in (ax_t, ax_r):
        ax.grid(alpha=0.25, which='major')
        ax.tick_params(which='both', direction='in')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--dataset', default='GD89',
                        help="sublimation dataset (default GD89, the kinetics the "
                             "literature relations assume; 'fiducial' for the "
                             "laboratory data)")
    args = parser.parse_args()
    sk.set_default_dataset(args.dataset)
    print(f"Sublimation dataset: {args.dataset}")
    print_comparison()
    if not args.no_plot:
        out_dir = _paths.get_plots_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        plot(out_dir / 'sublimation_radius_agn.png')


if __name__ == '__main__':
    main()
