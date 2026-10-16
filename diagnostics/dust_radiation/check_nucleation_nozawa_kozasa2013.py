#!/usr/bin/env python3
"""Nucleation check: reproduce Nozawa & Kozasa (2013) for carbon grains.

Nozawa & Kozasa (2013, ApJ 776, 24; NK13) follow dust formation in gas
cooling as it expands homologously, c~(t) = c10 (t/t0)^-3 and
T = T0 (t/t0)^-3(gamma-1) (their eqs. 54-55), with t0 = 300 d the time at
which ln S = 0 (eq. 56). The final average radius and condensation
efficiency depend only on Lambda_on = tau_sat/tau_coll at the onset of
formation (f_con = 1e-10; eqs. 57, 58, 63), and they fit them as

    log(a_ave,inf / a* - 1) = eps1 + eps2 log Lambda_on            (eq. 64)
    log f_con,inf = chi1 [tanh(chi2 log Lambda_on + chi3) - 1]       (eq. 65)

with Table 1 (C: a* = 5.946e-4 um, eps1 = -1.21, eps2 = 0.854, chi1 = 4.15,
chi2 = 1.91, chi3 = 0.088), to 5% in a_ave for Lambda_on <= 1e6. Their
steady-state model (eq. 59) follows these fits for Lambda_on >~ 30.

This script runs that steady model with pyCALIMA's
``dust_sublimation.classical_nucleation_rate`` and NK13's thermodynamics
for C (Nozawa et al. 2003, ApJ 598, 785, Table 2: Delta G/kT = -A/T + B,
A = 8.64726e4 K, B = 19.0422; sigma = 1400 erg cm^-2; a0 = 1.281 A;
sticking s = 1):

    ln S  = A/T - B + ln(c1 k T / p_s),   p_s = 1 bar           (eq. 56)
    I_s   = J_s / c~1,   n_crit = 1 + (2 mu / 3 ln S)^3          (A9)
    da/dt = s Omega_0 (k T / 2 pi m_1)^1/2 c1 (1 - 1/S)          (eq. 2)
    dK_i/dt = I_s n_crit^(i/3) + (i/a0)(da/dt) K_(i-1)          (eq. 49)
    c1 = c~1 (1 - K3),   f_con = K3,   a_ave = a0 (K3/K0)^(1/3)

Usage
-----
    python diagnostics/dust_radiation/check_nucleation_nozawa_kozasa2013.py
"""
import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pycalima import _paths
from pycalima.models.dust_radiation.dust_sublimation import classical_nucleation_rate

KB = 1.380649e-16
AMU = 1.66053906660e-24
DAY = 86400.0
P_S = 1.0e6                    # 1 bar [dyn cm^-2]

# Nozawa et al. (2003) Table 2, C(s)
A_C, B_C = 8.64726e4, 19.0422
SIGMA_C = 1400.0               # [erg cm^-2]
A0_C = 1.281e-8                # [cm]
M_C = 12.011 * AMU
# Nozawa & Kozasa (2013) Table 1, C
A_STAR_UM = 5.946e-4
EPS1, EPS2 = -1.21, 0.854
CHI1, CHI2, CHI3 = 4.15, 1.91, 0.088
T0_DAYS = 300.0


def nk13_a_ave(lam):
    return A_STAR_UM * (1.0 + 10 ** (EPS1 + EPS2 * np.log10(lam)))


def nk13_f_con(lam):
    return 10 ** (CHI1 * (np.tanh(CHI2 * np.log10(lam) + CHI3) - 1.0))


def ln_supersaturation(c1, T):
    return A_C / T - B_C + np.log(c1 * KB * T / P_S)


def run(c10, gamma, t0=T0_DAYS * DAY, s=1.0, x_end=20.0):
    """Steady model of NK13 for one (c10, gamma). Returns Lambda_on,
    a_ave,inf [um], f_con,inf, the onset time x_on = t_on/t0, tau_sat at the
    onset [days] and T0 [K]."""
    T0 = brentq(lambda T: ln_supersaturation(c10, T), 300.0, 1e4)
    omega0 = 4.0 * np.pi * A0_C ** 3 / 3.0
    vfac = np.sqrt(KB / (2.0 * np.pi * M_C))

    def gas(t, K3):
        x = t / t0
        c_nom = c10 * x ** -3
        T = T0 * x ** (-3.0 * (gamma - 1.0))
        c1 = c_nom * max(1.0 - K3, 0.0)
        return c_nom, c1, T

    def rhs(t, K):
        c_nom, c1, T = gas(t, K[3])
        if c1 <= 0.0:
            return np.zeros(4)
        lnS = ln_supersaturation(c1, T)
        J, n_star = classical_nucleation_rate(c1, lnS, T, T, A0_C, SIGMA_C, M_C, alpha=s)
        I_s = J / c_nom
        n_crit = 1.0 + n_star if np.isfinite(n_star) else 0.0
        dadt = s * omega0 * vfac * np.sqrt(T) * c1 * (1.0 - np.exp(-lnS)) if lnS > 0 else 0.0
        g = dadt / A0_C
        return np.array([I_s,
                         I_s * n_crit ** (1 / 3) + g * K[0],
                         I_s * n_crit ** (2 / 3) + 2 * g * K[1],
                         I_s * n_crit + 3 * g * K[2]])

    onset = lambda t, K: K[3] - 1e-10
    onset.terminal = False
    sol = solve_ivp(rhs, (t0, x_end * t0), np.zeros(4), method="LSODA",
                    rtol=1e-8, atol=1e-40, events=onset, dense_output=True)
    if not sol.t_events[0].size:
        return (np.nan,) * 6
    t_on = sol.t_events[0][0]
    c_nom, _, T_on = gas(t_on, 0.0)
    tau_sat_inv = 3.0 * (gamma - 1.0) / t_on * (A_C / T_on - gamma / (gamma - 1.0))
    tau_coll_inv = s * 4.0 * np.pi * A0_C ** 2 * c_nom * vfac * np.sqrt(T_on)
    lam = tau_coll_inv / tau_sat_inv
    K = sol.y[:, -1]
    return (lam, A0_C * (K[3] / K[0]) ** (1.0 / 3.0) * 1e4, K[3], t_on / t0,
            1.0 / tau_sat_inv / DAY, T0)


def compute(c10_grid, gammas):
    rows = []
    for gamma in gammas:
        for c10 in c10_grid:
            lam, a_ave, f_con, x_on, tau_sat_d, T0 = run(c10, gamma)
            rows.append((gamma, c10, lam, a_ave, f_con, x_on, tau_sat_d, T0))
    return rows


def print_comparison(rows):
    print("Nozawa & Kozasa (2013) steady model for C grains, t0 = 300 d")
    print(f"{'gamma':>6s}{'c10':>10s}{'Lambda_on':>11s}{'a_ave [um]':>12s}{'NK13 eq64':>11s}"
          f"{'ratio':>7s}{'f_con':>9s}{'NK13 eq65':>11s}")
    for gamma, c10, lam, a, f, *_ in rows:
        print(f"{gamma:6.2f}{c10:10.1e}{lam:11.3g}{a:12.4g}{nk13_a_ave(lam):11.4g}"
              f"{a / nk13_a_ave(lam):7.3f}{f:9.3g}{nk13_f_con(lam):11.3g}")
    ok = [r for r in rows if r[2] >= 30]
    dev = max(abs(r[3] / nk13_a_ave(r[2]) - 1) for r in ok)
    print(f"max |a_ave / eq64 - 1| for Lambda_on >= 30: {dev:.1%} (NK13 quote 5% for their fit)")
    lam, a, f, x_on, tau_sat_d, T0 = run(1e8, 1.25)
    print(f"c10 = 1e8, gamma = 1.25: T0 = {T0:.0f} K (NK13 Fig. 1: ~2000 K), "
          f"x_on = {x_on:.3f} (non-steady NK13: 1.08), tau_sat = {tau_sat_d:.1f} d "
          f"(NK13: ~10 d)")


def plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_f) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    lam = np.logspace(0.5, 6.5, 200)
    ax_a.plot(lam, nk13_a_ave(lam), color="#52514e", lw=1.5, label="NK13 eq. 64")
    ax_f.plot(lam, nk13_f_con(lam), color="#52514e", lw=1.5, label="NK13 eq. 65")
    for gamma, marker, color in ((1.25, "o", "#2a78d6"), (1.5, "s", "#eb6834")):
        sel = [r for r in rows if r[0] == gamma]
        ax_a.plot([r[2] for r in sel], [r[3] for r in sel], marker, color=color, ms=6,
                  label=rf"pyCALIMA, $\gamma$ = {gamma}")
        ax_f.plot([r[2] for r in sel], [r[4] for r in sel], marker, color=color, ms=6)
    for ax in (ax_a, ax_f):
        ax.axvspan(lam[0], 30, color="#52514e", alpha=0.08, lw=0)
        ax.set(xscale="log", xlabel=r"$\Lambda_{\rm on}=\tau_{\rm sat}/\tau_{\rm coll}$")
        ax.tick_params(which="both", direction="in", top=True, right=True)
    ax_a.text(4, 0.3, "steady state\nnot valid\n(NK13)", fontsize=8, color="#52514e")
    ax_a.set(yscale="log", ylabel=r"$a_{\rm ave,\infty}$ [$\mu$m]")
    ax_f.set(yscale="log", ylabel=r"$f_{\rm con,\infty}$", ylim=(1e-4, 2))
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")
    ax_f.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("C grains in cooling gas: pyCALIMA steady-state nucleation vs "
                 "Nozawa & Kozasa (2013)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Saved", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    rows = compute(np.logspace(5.0, 10.5, 12), (1.25, 1.5))
    print_comparison(rows)
    if not args.no_plot:
        out_dir = _paths.get_plots_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        plot(rows, out_dir / "nucleation_nozawa_kozasa2013.png")


if __name__ == "__main__":
    main()
