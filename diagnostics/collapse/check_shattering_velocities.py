"""Can grains shatter in a collapsing low-metallicity cloud?

Largest rms relative velocity between two grains of radii 1 nm - 1 um (Brownian
plus Ormel & Cuzzi 2007 turbulence driven at the Jeans length with v_g = Mach c_s,
the kernel of ``integrator.turbulent_kernel``) along the Omukai+05 Z = 0 track,
against the shattering thresholds of Jones, Tielens & Hollenbach (1996, Table 1):
1.2 km/s (carbon) and 2.7 km/s (silicates). Ambipolar drift is left out (the
gas is nearly neutral at these densities). Also printed: the number of
collisions above the carbon threshold that a 0.1 um grain suffers during the
whole collapse, with the N07 20 Msun CCSN dust at [Z] = -3.5 (the top of the
[Z_cr] range): sum_j n_j pi (a + a_j)^2 dv over the collapse time, and the
projectile mass so delivered per unit target mass (sum_j hits_j m_j / m): small
projectiles crater rather than disrupt, ejecting of order (v / v_shat)^2 times
their own mass (Jones et al. 1996), so a value << 1 leaves the target intact.

    python diagnostics/collapse/check_shattering_velocities.py

Writes ``collapse/shattering_velocities.png`` under the plots directory.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.collapse import velocities as vel
from pycalima.collapse.tracks import Track
from pycalima.plotting_style import use_calima_style

V_SHAT = {"carbon": 1.2e5, "silicate": 2.7e5}       # cm/s, Jones et al. (1996)
RHO_S, SIGMA_N = 3.0, 2.0e-15


def max_dv(track, nH, mach, a=np.geomspace(1e-7, 1e-4, 120)):
    T, mu, rho = float(track.T(nH)), float(track.mu(nH)), float(track.rho(nH))
    v_g = mach * vel.sound_speed(T, mu)
    L = vel.jeans_length(T, rho, mu)
    tau_L = L / v_g
    tau_eta = tau_L / math.sqrt(max(v_g * L / (vel.thermal_speed(T, mu) / (3 * nH * SIGMA_N)), 1.0))
    m = 4.0 / 3.0 * math.pi * RHO_S * a ** 3
    t_s = vel.epstein_stopping_time(a, RHO_S, T, rho, mu)
    dv = np.sqrt(vel.brownian_rms(T, m) ** 2 + vel.ormel_cuzzi_dv(t_s, v_g, tau_L, tau_eta) ** 2)   # pairs
    i, j = np.unravel_index(dv.argmax(), dv.shape)
    return dv.max(), (a[i], a[j])


def hits_above_threshold(track, mach, v_shat=V_SHAT["carbon"], a_target=1e-5, log_z=-3.5):
    """Collisions above v_shat per a_target grain over the collapse (free fall, d ln n = 0.05)."""
    from pycalima.collapse.ics import nozawa_ccsn
    pops = nozawa_ccsn(20, 10.0 ** log_z)[0]
    g = pops[0].grid
    a = np.sqrt(g.a_edges[1:] * g.a_edges[:-1])
    per_H = sum(p.x / (4.0 / 3.0 * math.pi * p.grid.rho_s * a ** 3) for p in pops)      # grains per H per bin
    a = np.append(a, a_target)
    per_H = np.append(per_H, 0.0)
    n = np.logspace(-1, 16, 341)
    total, mass = 0.0, 0.0
    for x, x1 in zip(n[:-1], n[1:]):
        nm = math.sqrt(x * x1)
        T, mu, rho = float(track.T(nm)), float(track.mu(nm)), float(track.rho(nm))
        v_g = mach * vel.sound_speed(T, mu)
        L = vel.jeans_length(T, rho, mu)
        tau_eta = (L / v_g) / math.sqrt(max(v_g * L / (vel.thermal_speed(T, mu) / (3 * nm * SIGMA_N)), 1.0))
        m = 4.0 / 3.0 * math.pi * RHO_S * a ** 3
        dv = np.sqrt(vel.brownian_rms(T, m) ** 2
                     + vel.ormel_cuzzi_dv(vel.epstein_stopping_time(a, RHO_S, T, rho, mu), v_g, L / v_g, tau_eta) ** 2)[-1]
        hits = per_H * nm * math.pi * (a_target + a) ** 2 * dv * (dv > v_shat) * float(track.t_col(nm)) * math.log(x1 / x)
        total += hits.sum()
        mass += np.sum(hits * (a / a_target) ** 3 * (dv / v_shat) ** 2)
    return total, mass


def main():
    use_calima_style()
    tr = Track.omukai2005("Z0")
    n = np.logspace(0, 16, 81)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mach, c in ((1, "C0"), (3, "C1"), (10, "C3")):
        v = np.array([max_dv(tr, x, mach)[0] for x in n])
        ax.loglog(n, v / 1e5, color=c, label=f"Mach {mach}")
        print(f"Mach {mach}: max over n of the largest dv = {v.max() / 1e5:.3f} km/s at n_H = {n[v.argmax()]:.1e}; "
              "collisions above 1.2 km/s per 0.1 um grain over the collapse: %.2e, ejected mass / target mass ~ %.2e"
              % hits_above_threshold(tr, mach))
    for k, v in V_SHAT.items():
        ax.axhline(v / 1e5, color="k", ls="--" if k == "carbon" else ":", lw=1, label=f"shattering, {k} (Jones+96)")
    ax.set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"largest grain-grain $\Delta v$ [km s$^{-1}$]",
           title="Relative velocities along the Omukai+05 Z = 0 collapse (1 nm - 1 um)")
    ax.legend(fontsize=8)
    out = _paths.get_plots_dir() / "collapse"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "shattering_velocities.png", dpi=150)
    print(f"wrote {out / 'shattering_velocities.png'}")


if __name__ == "__main__":
    main()
