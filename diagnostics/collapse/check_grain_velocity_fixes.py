#!/usr/bin/env python3
"""Grain relative velocities before and after the October 2026 physics fixes.

``solvers.grain_dynamics._ormel2007`` (and RAMSES ``dust_dynamics.f90``,
function ``grain_relative_velocity``, model 'Ormel2007') had three errors:

1. Brownian term sqrt(8 k T (m1 + m2) / (m1 m2)): missing pi (x 1.77 too fast).
2. Gas thermal speed for the Epstein stopping time sqrt(8 / pi) c_s, with
   c_s = sqrt(5/3 k T / mu m_H): the adiabatic index does not belong in it
   (t_s 1.29x too short, turbulent dv of intermediate-regime grains 1.14x too low).
3. Reynolds number from a Coulomb mean free path 1 / (n_H r_c^2) whatever the
   ionisation: in neutral gas at 10-100 K this makes Re ~ 1e20-1e24 and puts
   every grain in the intermediate regime. Now the viscosity of the
   neutral + ionised mixture (mass-weighted, Braginskii for the ions).

4. RAMSES never orders target and projectile, so its regime choice uses the
   target's stopping time even when the projectile is the larger grain (NaN in
   the tightly coupled branch); the Python mirror always did.
5. The turbulent term switched between the three limit formulas of Ormel &
   Cuzzi (their Eqs. 27-29), which jump by up to x2.4 at t_eta and t_L. Now
   their full closed form (Eqs. 16-18, boundary eddy t* = 1.6 t_s): continuous;
   the remaining kinks are physical turning points (class II onset, t_s = t_L).

Writes ``velocity_fixes.png`` to ``_paths.get_plots_dir("collapse")``: the
relative velocity of a 0.1 um grain with partners of 1 nm - 10 um, before and
after, in four environments.

    python diagnostics/collapse/check_grain_velocity_fixes.py
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.plotting_style import use_calima_style
from pycalima.solvers.grain_dynamics import grain_relative_velocity

KB, MH, E2 = 1.3806488e-16, 1.6726219e-24, 2.3070779e-19


def ramses_prefix(T, rho, nH, v_turb, mu, L, a1, s1, m1, a2, s2, m2):
    """The pre-fix formulae, as in RAMSES dust_dynamics.f90 (with the target /
    projectile ordering of the Python mirror, so that it stays finite), including
    the switch between the three limit formulas."""
    dV_th = math.sqrt(8 * KB * T * (m1 + m2) / (m1 * m2))
    cs = math.sqrt(5 / 3 * KB * T / (MH * mu))
    v_th = math.sqrt(8 / math.pi) * cs
    mfp = 1 / (nH * (E2 / (KB * T)) ** 2)
    tau_L = L / v_turb
    Re = max(3 * v_turb * L / (cs * mfp), 1.0)
    tau_eta = tau_L / math.sqrt(Re)
    t1, t2 = sorted((s1 * a1 / (rho * v_th), s2 * a2 / (rho * v_th)), reverse=True)
    S1, S2, Smin = t1 / tau_L, t2 / tau_L, tau_eta / tau_L
    if t1 < tau_eta:
        arg = (S1 - S2) / (S1 + S2) * (S1 ** 2 / (S1 + Smin) - S2 ** 2 / (S2 + Smin))
    elif t1 < tau_L:
        x = S2 / S1
        arg = (3.2 - (1 + x) + 2 / (1 + x) * (1 / 2.6 + x ** 3 / (1.6 + x))) * S1
    else:
        arg = 1 / (1 + S1) + 1 / (1 + S2)
    return math.sqrt(dV_th ** 2 + 1.5 * v_turb ** 2 * max(arg, 0.0))


def main():
    use_calima_style()
    plt.rcParams["text.usetex"] = False
    envs = [("WIM: $n=0.1$, 8000 K, $x=1$", 0.1, 8000.0, 1.0, 1e6),
            ("CNM: $n=30$, 80 K, $x=10^{-3}$", 30.0, 80.0, 1e-3, 1e5),
            ("MC: $n=10^3$, 20 K, $x=10^{-7}$", 1e3, 20.0, 1e-7, 3e4),
            ("core: $n=10^6$, 10 K, $x=10^{-8}$", 1e6, 10.0, 1e-8, 2e4)]
    L, mu, s = 10 * 3.086e18, 1.4, 3.0
    a1 = 1e-5
    m1 = 4 / 3 * math.pi * s * a1 ** 3
    a2 = np.geomspace(1e-7, 1e-3, 200)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (name, n, T, x, vt) in zip(axes, envs):
        rho = mu * MH * n
        old, new = [], []
        for a in a2:
            m2 = 4 / 3 * math.pi * s * a ** 3
            old.append(ramses_prefix(T, rho, n, vt, mu, L, a1, s, m1, a, s, m2))
            new.append(grain_relative_velocity("Ormel2007", T, rho, n, vt, mu, L, a1, s, m1, a, s, m2, x_ion=x))
        ax.plot(a2 * 1e4, np.array(old) / 1e5, "--", color="tab:red", label="before (RAMSES formulae)")
        ax.plot(a2 * 1e4, np.array(new) / 1e5, "-", color="k", label="after (fixed)")
        ax.axvline(a1 * 1e4, color="0.7", lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel=r"partner radius [$\mu$m]", title=name)
    axes[0].set_ylabel(r"$\Delta v$ with a 0.1 $\mu$m grain [km s$^{-1}$]")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Ormel & Cuzzi (2007) relative velocities, $L$ = 10 pc, $\mu$ = 1.4: before and after the fixes")
    fig.tight_layout()
    out = _paths.get_plots_dir("collapse") / "velocity_fixes.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
