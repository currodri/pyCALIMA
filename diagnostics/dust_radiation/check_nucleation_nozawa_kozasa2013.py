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
``dust_sublimation.classical_nucleation_rate``, and their non-steady model
(cluster rate equations for n = 2 .. 99 with n* = 100; eqs. 7, 15, 39,
44-45, 49), whose results eqs. 64-65 fit (their steady model keeps
f_con = 1 down to Lambda_on ~ 0.3, the grey lines of their Fig. 8), with
NK13's thermodynamics
for C (Nozawa et al. 2003, ApJ 598, 785, Table 2: Delta G/kT = -A/T + B,
A = 8.64726e4 K, B = 19.0422; sigma = 1400 erg cm^-2; a0 = 1.281 A;
sticking s = 1):

    ln S  = A/T - B + ln(c1 k T / p_s),   p_s = 1 bar           (eq. 56)
    I_s   = J_s / c~1,   n_crit = 1 + (2 mu / 3 ln S)^3          (A9)
    da/dt = s Omega_0 (k T / 2 pi m_1)^1/2 c1 (1 - 1/S)          (eq. 2)
    dK_i/dt = I_s n_crit^(i/3) + (i/a0)(da/dt) K_(i-1)          (eq. 49)
    c1 = c~1 (1 - K3),   f_con = K3,   a_ave = a0 (K3/K0)^(1/3)

Finally the same expansion is run through the grain-size-distribution
solver (``pycalima.gsd``: exchange steps plus dilution, steady-state
nucleation injected past the barrier, Kelvin term in growth), with the
N03 graphite dataset set to NK13's s = 1 and a0 (``run_gsd``).

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


N_STAR = 100        # smallest grain, in monomers (NK13 s. 3.3)


def run_nonsteady(c10, gamma, t0=T0_DAYS * DAY, s=1.0, x_end=20.0, n_star=N_STAR):
    """Non-steady model of NK13 for C (single-element grain): rate equations
    for clusters n = 2 .. n*-1 (eqs. 7, 39, 44, 45), grains formed by the
    current into n* (closure eq. 15) and grown by eq. 2 (moments, eq. 49).
    Returns the same tuple as ``run``."""
    T0 = brentq(lambda T: ln_supersaturation(c10, T), 300.0, 1e4)
    omega0 = 4.0 * np.pi * A0_C ** 3 / 3.0
    n = np.arange(1, n_star + 1, dtype=float)                 # 1 .. n*
    # alpha_n / sqrt(T): attachment of a monomer to an n-mer (eq. 7)
    m_red = n * M_C / (n + 1.0)
    alpha_red = (s / np.where(n == 1, 2.0, 1.0) * 4.0 * np.pi * A0_C ** 2 * n ** (2 / 3)
                 * np.sqrt(KB / (2.0 * np.pi * m_red)))
    dn23 = (n - 1.0) ** (2 / 3) - np.maximum(n - 2.0, 0.0) ** (2 / 3)   # gamma_n bracket, n>=2
    nc = np.arange(2, n_star)                                  # cluster sizes kept
    vfac = np.sqrt(KB / (2.0 * np.pi * M_C))

    def state(t, y):
        x = t / t0
        c_nom = c10 * x ** -3
        T = T0 * x ** (-3.0 * (gamma - 1.0))
        Yc, K = y[:nc.size], y[nc.size:]
        Y1 = 1.0 - np.sum(nc * Yc) - K[3]
        return c_nom, T, max(Y1, 0.0), Yc, K

    def rhs(t, y):
        c_nom, T, Y1, Yc, K = state(t, y)
        c1 = Y1 * c_nom
        if c1 <= 0.0:
            return np.zeros_like(y)
        lnS = ln_supersaturation(c1, T)
        mu = 4.0 * np.pi * A0_C ** 2 * SIGMA_C / (KB * T)
        gam = mu * dn23 - lnS                       # gamma_n, index n-1
        rate = alpha_red * np.sqrt(T) * c1          # tau_(n)^-1 = alpha_n c1, index n-1
        Yall = np.concatenate([[Y1], Yc])           # Y_1 .. Y_(n*-1)
        # I_n for n = 2 .. n*-1:  rate_(n-1) [Y_(n-1) - Y_n exp(gamma_n)]
        I = rate[:n_star - 2] * (Yall[:-1] - Yc * np.exp(np.minimum(gam[1:n_star - 1], 700)))
        # closure: I* = I_(n*) = rate_(n*-1) Y_(n*-1) [1 - exp(gamma_(n*))]
        I_star = rate[n_star - 2] * Yc[-1] * (1.0 - np.exp(min(gam[n_star - 1], 700.0)))
        dY = I - np.append(I[1:], I_star)
        dadt = (s * omega0 * vfac * np.sqrt(T) * c1 * (1.0 - np.exp(-lnS))) if lnS > 0 else 0.0
        g = dadt / A0_C
        dK = [I_star, I_star * n_star ** (1 / 3) + g * K[0],
              I_star * n_star ** (2 / 3) + 2 * g * K[1], I_star * n_star + 3 * g * K[2]]
        return np.concatenate([dY, dK])

    # initial clusters: equilibrium abundances at ln S = 0 (NK13 s. 3.3)
    mu0 = 4.0 * np.pi * A0_C ** 2 * SIGMA_C / (KB * T0)
    Y0 = np.exp(-np.cumsum(mu0 * dn23[1:n_star - 1]))
    y0 = np.concatenate([Y0, np.zeros(4)])
    onset = lambda t, y: y[-1] - 1e-10
    sol = solve_ivp(rhs, (t0, x_end * t0), y0, method="LSODA", rtol=1e-8, atol=1e-40,
                    events=onset)
    if not sol.t_events[0].size:
        return (np.nan,) * 6
    t_on = sol.t_events[0][0]
    c_nom, T_on, _, _, _ = state(t_on, sol.y_events[0][0])
    tau_sat_inv = 3.0 * (gamma - 1.0) / t_on * (A_C / T_on - gamma / (gamma - 1.0))
    tau_coll_inv = s * 4.0 * np.pi * A0_C ** 2 * c_nom * vfac * np.sqrt(T_on)
    K = sol.y[-4:, -1]
    return (tau_coll_inv / tau_sat_inv, A0_C * (K[3] / K[0]) ** (1.0 / 3.0) * 1e4, K[3],
            t_on / t0, 1.0 / tau_sat_inv / DAY, T0)


def nk13_carbon_model():
    """N03 graphite with NK13's sticking (s = 1) and a0 = 1.281 A per atom."""
    from dataclasses import replace

    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    m = get_sublimation_model("graphite", "N03")
    rho = M_C / (4.0 * np.pi * A0_C ** 3 / 3.0)
    return replace(m, channels=(replace(m.channels[0], alpha0=1.0),), density_gcm3=rho)


def run_gsd(c10, gamma, t0=T0_DAYS * DAY, x_end=20.0, step=5e-4, kelvin=True,
            bins_per_mass_decade=10):
    """NK13's expansion through pycalima.gsd. Returns the tuple of ``run``
    (Lambda_on, x_on, tau_sat and T0 from the steady model).

    The step dt = ``step`` t must resolve the cooling: within a step T is
    held fixed and ln S moves by (A/T) dln T. At gamma = 1.5 and Lambda_on
    ~ 1e5-1e6, step = 2e-3 puts a_ave 20-50% high; 5e-4 is converged to ~1%.
    """
    from pycalima.gsd.exchange import Population, exchange_step

    model = nk13_carbon_model()
    lam, _, _, x_on, tau_sat, T0 = run(c10, gamma, t0=t0)
    pop = Population.empty(model, 0.9e-8, 1e-2, bins_per_mass_decade, T_d=T0, kelvin=kelvin)
    gas, t = {"C": c10}, t0
    while t < x_end * t0:
        dt = step * t
        d = (t / (t + dt)) ** 3                  # homologous dilution
        t += dt
        pop.N, pop.M, gas["C"] = pop.N * d, pop.M * d, gas["C"] * d
        pop.overflow_N *= d
        pop.overflow_M *= d
        T = T0 * (t / t0) ** (-3.0 * (gamma - 1.0))
        pop.T_d = np.full(pop.grid.n_bins, T)
        exchange_step([pop], gas, T, dt)
    f = pop.key_atoms() / (c10 * (t / t0) ** -3)
    n = pop.N.sum() + pop.overflow_N
    a = ((pop.M.sum() + pop.overflow_M) / n / (4.0 / 3.0 * np.pi * model.density_gcm3)) ** (1 / 3)
    return lam, a * 1e4, f, x_on, tau_sat, T0


def compute(c10_grid, gammas):
    rows = []
    for gamma in gammas:
        for c10 in c10_grid:
            rows.append(("steady", gamma, c10) + tuple(run(c10, gamma)))
            rows.append(("GSD", gamma, c10) + tuple(run_gsd(c10, gamma)))
            # Above c10 ~ 1e9 (Lambda_on >~ 1e5) the cluster equations become very
            # stiff (tau_coll ~ 0.03 s against t0 = 2.6e7 s) and the steady model,
            # which NK13 show is exact there, is used instead.
            if c10 <= 1.0e9:
                rows.append(("non-steady", gamma, c10) + tuple(run_nonsteady(c10, gamma)))
    return rows


def print_comparison(rows):
    print("Nozawa & Kozasa (2013), C grains, t0 = 300 d. eqs. 64-65 fit their NON-STEADY")
    print("model; their steady model keeps f_con = 1 down to Lambda_on ~ 0.3 (Fig. 8, grey).")
    print(f"{'model':>11s}{'gamma':>6s}{'c10':>9s}{'Lambda_on':>11s}{'a_ave [um]':>12s}"
          f"{'eq64':>10s}{'ratio':>7s}{'f_con':>9s}{'eq65':>9s}")
    for model, gamma, c10, lam, a, f, *_ in rows:
        print(f"{model:>11s}{gamma:6.2f}{c10:9.1e}{lam:11.3g}{a:12.4g}{nk13_a_ave(lam):10.4g}"
              f"{a / nk13_a_ave(lam):7.3f}{f:9.3g}{nk13_f_con(lam):9.3g}")
    ns = [r for r in rows if r[0] == "non-steady" and np.isfinite(r[3]) and r[3] <= 1e6]
    dev_a = max(abs(r[4] / nk13_a_ave(r[3]) - 1) for r in ns)
    dev_f = max(abs(np.log10(r[5] / nk13_f_con(r[3]))) for r in ns if r[5] > 1e-3)
    print(f"non-steady vs fits, Lambda_on <= 1e6: max |a/eq64 - 1| = {dev_a:.1%}, "
          f"max |log f/eq65| (f > 1e-3) = {dev_f:.2f} dex")
    st = [r for r in rows if r[0] == "steady" and np.isfinite(r[3]) and r[3] >= 30]
    print(f"steady vs eq64, Lambda_on >= 30: max |a/eq64 - 1| = "
          f"{max(abs(r[4] / nk13_a_ave(r[3]) - 1) for r in st):.1%}")
    gs = [r for r in rows if r[0] == "GSD" and np.isfinite(r[3]) and r[3] >= 30]
    print(f"GSD solver vs eq64, Lambda_on >= 30: max |a/eq64 - 1| = "
          f"{max(abs(r[4] / nk13_a_ave(r[3]) - 1) for r in gs):.1%}")
    for label, fn in (("steady", run), ("non-steady", run_nonsteady)):
        lam, a, f, x_on, tau_sat_d, T0 = fn(1e5, 1.25)
        print(f"{label:>11s}, c10 = 1e5, gamma = 1.25: a_ave = {a:.4f} um, f_con = {f:.2f} "
              f"(NK13 s. 4.2: steady 0.0004 um, 1; non-steady 0.0007 um, ~0.3)")
    lam, a, f, x_on, tau_sat_d, T0 = run_nonsteady(1e8, 1.25)
    print(f"non-steady, c10 = 1e8, gamma = 1.25: T0 = {T0:.0f} K (NK13 Fig. 1: ~2000 K), "
          f"x_on = {x_on:.3f} (NK13: 1.08), tau_sat = {tau_sat_d:.1f} d (NK13: ~10 d)")


def plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_f) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    lam = np.logspace(-1, 6.5, 300)
    ax_a.plot(lam, nk13_a_ave(lam), color="#52514e", lw=1.5, label="NK13 eqs. 64-65 (non-steady fits)")
    ax_f.plot(lam, nk13_f_con(lam), color="#52514e", lw=1.5)
    for gamma, marker, color in ((1.25, "o", "#2a78d6"), (1.5, "s", "#eb6834")):
        for model, fill in (("non-steady", color), ("steady", "none")):
            sel = [r for r in rows if r[0] == model and r[1] == gamma]
            kw = dict(ls="none", marker=marker, ms=6, mec=color, mfc=fill)
            ax_a.plot([r[3] for r in sel], [r[4] for r in sel],
                      label=rf"pyCALIMA {model}, $\gamma$ = {gamma}", **kw)
            ax_f.plot([r[3] for r in sel], [r[5] for r in sel], **kw)
        sel = [r for r in rows if r[0] == "GSD" and r[1] == gamma]
        kw = dict(ls="none", marker="x", ms=7, mew=1.5, color="#1c1c1a" if gamma == 1.25 else "#8a3ffc")
        ax_a.plot([r[3] for r in sel], [r[4] for r in sel],
                  label=rf"pycalima.gsd solver, $\gamma$ = {gamma}", **kw)
        ax_f.plot([r[3] for r in sel], [r[5] for r in sel], **kw)
    for ax in (ax_a, ax_f):
        ax.set(xscale="log", xlabel=r"$\Lambda_{\rm on}=\tau_{\rm sat}/\tau_{\rm coll}$",
               xlim=(0.1, 3e6))
        ax.tick_params(which="both", direction="in", top=True, right=True)
    ax_a.set(yscale="log", ylabel=r"$a_{\rm ave,\infty}$ [$\mu$m]")
    ax_f.set(yscale="log", ylabel=r"$f_{\rm con,\infty}$", ylim=(1e-4, 2))
    ax_a.legend(frameon=False, fontsize=7, loc="upper left")
    ax_f.text(0.5, 3e-4, "open: steady model (NK13 Fig. 8, grey)\nfilled: non-steady\ncrosses: GSD solver (steady kernel)",
              fontsize=8, color="#52514e")
    fig.suptitle("C grains in cooling gas: pyCALIMA vs Nozawa & Kozasa (2013)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Saved", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    rows = compute(np.logspace(3.0, 10.5, 16), (1.25, 1.5))
    print_comparison(rows)
    if not args.no_plot:
        out_dir = _paths.get_plots_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        plot(rows, out_dir / "nucleation_nozawa_kozasa2013.png")


if __name__ == "__main__":
    main()
