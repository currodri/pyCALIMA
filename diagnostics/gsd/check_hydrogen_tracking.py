"""Tabulate or track the hydrogen of a-C(:H) grains? P2 tests T1, T4, T5, T6, grain half.

The covalent bins (1-53 nm) carry H/C under the laboratory laws of ``gsd.hydrogen`` (UV
loss Y_CH = 0.02 per absorbed photon down to X_H = 0.05; Mennella's H addition up to bulk
H/C 0.65). For bins 4-7 of the default ``CarbonLayout`` this prints:

- T1: the relaxation time 1/(k_UV + k_H) at equilibrium and the equilibrium band gap on
  a grid of FUV (Mathis units) and n(H I) at 20, 100 and 1000 K, with a figure;
- T4/T6: parcels carried through the Orion Bar (isobaric) and Horsehead profiles of
  Murga+20 at 0.3, 1 and 3 km/s, into and out of the cloud, and a parcel cycling between
  CNM and WNM: the true H/C (implicit ODE solution) against the equilibrium the tables
  would give and against RAMSES-like steps of 1e2, 1e3 and 1e4 yr (conditions held per
  step, the exact step with frozen rates, ``BinHydrogen.step``). Errors are the largest
  along the path, in E_g and in ln kappa at 3.4, 0.55 and 100 um;
- T5: the bias of the tracked moment when two parcels mix: kappa at the mean H/C
  against the mean kappa.

Needs the Murga+20 Figshare profiles (dataset murga2020-figshare) and the tables of
``diagnostics/gsd/check_hydrogenated_tables.py`` (``--tables``).

    python diagnostics/gsd/check_hydrogen_tracking.py [--murga DIR] [--tables FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pycalima.gsd.carbon import CarbonLayout
from pycalima.gsd.hydrogen import YR, BinHydrogen, Conditions, k_h
from pycalima.gsd.optics import lookup_x

BINS = (4, 5, 6, 7)
LAM_PROBE_UM = (3.4, 0.55, 100.0)
SPEEDS_KMS = (0.3, 1.0, 3.0)
STEPS_YR = (1e2, 1e3, 1e4)
NH_PER_AV = 1.87e21                 # cm^-2 mag^-1
CNM = Conditions(1.0, 50.0, 80.0)
WNM = Conditions(1.0, 0.5, 8000.0)


def load_tables(path):
    d = np.load(path)
    lam = d["lam_um"]
    il = [int(np.argmin(abs(lam - p))) for p in LAM_PROBE_UM]
    return {b: (d[f"bin{b}_s"], d[f"bin{b}_c"], d[f"bin{b}_kappa"][:, :, il, 0]) for b in BINS}


def ln_kappa(tab, e_gap, s=0.5):
    s_t, e_t, k = tab
    return np.log(lookup_x(s_t, e_t, k, s, float(e_gap)))


def profile_path(site, root, speed_kms, inward):
    """Conditions against time [s] for a parcel crossing the Murga+20 profile."""
    from pycalima.pahchem.murga_data import read_profile_2020

    p = read_profile_2020(site, root)
    av, n_tot, xh, xh2 = p["av"], p["n_tot"], p["x_h"], p["x_h2"]
    n_nuc = n_tot * (xh + 2 * xh2) / (xh + xh2)
    n_hi = n_tot * xh / (xh + xh2)
    dl = np.diff(av) * NH_PER_AV / (0.5 * (n_nuc[1:] + n_nuc[:-1]))
    t = np.concatenate([[0.0], np.cumsum(dl)]) / (speed_kms * 1e5)
    q = np.vstack([np.log(np.maximum(p["chi"], 1e-30)), np.log(np.maximum(n_hi, 1e-30)), np.log(p["t_gas"])])
    if not inward:
        t, q = t[-1] - t[::-1], q[:, ::-1]

    def cond(tt):
        v = [np.exp(np.interp(tt, t, qq)) for qq in q]
        return Conditions(v[0], v[1], v[2])
    return t[-1], cond


def cycle_path(period_yr, n_cycles=4):
    """CNM and WNM alternating every half period, with 2% of the period to switch."""
    P = period_yr * YR

    def cond(tt):
        ph = (tt % P) / P
        w = np.clip((ph - 0.5) / 0.02, 0.0, 1.0) * np.clip((1.0 - ph) / 0.02, 0.0, 1.0)
        mix = lambda a, b: np.exp((1 - w) * np.log(a) + w * np.log(b))          # noqa: E731
        return Conditions(mix(CNM.u_mathis, WNM.u_mathis), mix(CNM.n_hi, WNM.n_hi), mix(CNM.t_gas, WNM.t_gas))
    return n_cycles * P, cond


def _solve(bh, t_end, cond, x0, t_out):
    sol = solve_ivp(lambda t, y: [bh.rhs(y[0], cond(t))], (0.0, t_end), [x0], method="Radau",
                    t_eval=t_out, rtol=1e-7, atol=1e-10, max_step=t_end / 400)
    return sol.y[0]


def run_path(bh, tab, t_end, cond, x0=None, n_out=400):
    """On one path: the true H/C (conditions continuous), the equilibrium, and RAMSES-like
    steps (conditions held over each step, the exact step with frozen rates). Returns the
    largest errors, in E_g and ln kappa, of: the equilibrium in the UV-lit (U >= 1) and
    shielded (U < 1) parts; per step size, the step scheme against the exact solution
    under the same held conditions ("scheme"), and that solution against the continuous
    one ("held conditions", the sampling a cell-by-cell code makes anyway). The scheme
    is compared at the step ends, the only times it defines."""
    t_out = np.linspace(0.0, t_end, n_out)
    x0 = bh.equilibrium(cond(0.0)) if x0 is None else x0
    lit = np.array([cond(t).u_mathis >= 1.0 for t in t_out])

    def obs(x):
        e = bh.e_gap(np.asarray(x))
        return e, np.array([ln_kappa(tab, ee) for ee in e])

    def err(a, b, mask=None):
        m = np.ones(a[0].size, bool) if mask is None else mask
        if not m.any():
            return np.nan, np.full(len(LAM_PROBE_UM), np.nan)
        return np.max(abs(a[0][m] - b[0][m])), np.max(abs(a[1][m] - b[1][m]), axis=0)

    true = obs(_solve(bh, t_end, cond, x0, t_out))
    eq = obs(np.array([bh.equilibrium(cond(t)) for t in t_out]))
    out = {"eq. lit": err(eq, true, lit), "eq. shielded": err(eq, true, ~lit)}
    for dt_yr in STEPS_YR:
        n = max(int(np.ceil(t_end / (dt_yr * YR))), 1)
        h = t_end / n
        held = lambda t, h=h: cond(min(np.floor(t / h), n - 1) * h)       # noqa: E731
        x_held = _solve(bh, t_end, held, x0, t_out)
        x = [x0]
        for j in range(n):
            x.append(bh.step(x[-1], cond(j * h), h))
        t_s = np.minimum(h * np.arange(n + 1), t_end)   # the scheme exists only at the step ends
        x_held_s = _solve(bh, t_end, held, x0, t_s) if n > 1 else np.array([x0, x_held[-1]])
        out[f"{dt_yr:.0e} yr scheme"] = err(obs(np.array(x)), obs(x_held_s))
        out[f"{dt_yr:.0e} yr held conditions"] = err(obs(x_held), true)
    return out, (true[0].min(), true[0].max()), (t_out, true[0], eq[0])


def t1_maps(hyd, out_png):
    u = np.logspace(-3, 5, 33)
    n = np.logspace(-1, 5, 25)
    res = {}
    for b in (4, 7):
        bh = hyd[b]
        for T in (20.0, 100.0, 1000.0):
            tau = np.empty((u.size, n.size))
            eg = np.empty_like(tau)
            for i, uu in enumerate(u):
                for j, nn in enumerate(n):
                    c = Conditions(uu, nn, T)
                    xe = bh.equilibrium(c)
                    tau[i, j], eg[i, j] = bh.relaxation_time(xe, c) / YR, bh.e_gap(xe)
            res[(b, T)] = (tau, eg)
    print("\nT1: relaxation time [yr] at equilibrium, bin 4 (1.06-2.82 nm), T = 100 K; rows FUV [Mathis], "
          "columns n(H I) [cm^-3]")
    cols = [0, 6, 12, 18, 24]
    print(f"{'U':>9s} " + " ".join(f"{n[j]:>9.2g}" for j in cols))
    for i in range(0, u.size, 4):
        print(f"{u[i]:9.2g} " + " ".join(f"{res[(4, 100.0)][0][i, j]:9.2e}" for j in cols))
    print("equilibrium E_g [eV], same grid:")
    for i in range(0, u.size, 4):
        print(f"{u[i]:9.2g} " + " ".join(f"{res[(4, 100.0)][1][i, j]:9.3f}" for j in cols))
    plot_t1(u, n, res, out_png)


def plot_t1(u, n, res, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.4), sharex=True, sharey=True)
    for c, T in enumerate((20.0, 100.0, 1000.0)):
        tau, eg = res[(4, T)]
        ax = axs[0, c]
        cs = ax.contourf(n, u, np.log10(tau), levels=np.arange(-2, 9.1, 0.5), cmap="viridis")
        ln = ax.contour(n, u, np.log10(tau), levels=[2, 4, 6], colors="w", linewidths=1.2)
        ax.clabel(ln, fmt=lambda v: rf"$10^{{{int(v)}}}$ yr", fontsize=9)
        ax.set_title(rf"bin 4 (1.06--2.82 nm), $T$ = {T:.0f} K", fontsize=12)
        ax = axs[1, c]
        cs2 = ax.contourf(n, u, eg, levels=np.linspace(0.2, 1.7, 16), cmap="magma")
        ax.contour(n, u, eg, levels=[0.5, 1.0, 1.5], colors="w", linewidths=0.8)
    for ax in axs.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    for ax in axs[1]:
        ax.set_xlabel(r"$n$(H\,{\sc i}) [cm$^{-3}$]", fontsize=14)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"FUV [Mathis ISRF]", fontsize=14)
    fig.colorbar(cs, ax=axs[0].tolist(), label=r"$\log_{10}$ relaxation time [yr]")
    fig.colorbar(cs2, ax=axs[1].tolist(), label=r"equilibrium bulk $E_g$ [eV]")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


def plot_paths(bh, tab, murga, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.pahchem.murga_data import read_profile_2020
    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    cases = (("orion_isobaric", True, "Orion Bar, inward"), ("horsehead", True, "Horsehead, inward"),
             ("horsehead", False, "Horsehead, outward from the cloud"))
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex="col")
    for c, (site, inward, title) in enumerate(cases):
        av = read_profile_2020(site, murga)["av"]
        ax = axs[0, c]
        for v, ls in ((0.3, ":"), (1.0, "-"), (3.0, "--")):
            t_end, cond = profile_path(site, murga, v, inward)
            _, _, (t, e_true, e_eq) = run_path(bh, tab, t_end, cond, None if inward else bh.x_max)
            av_t = np.interp(t, *((np.linspace(0, t_end, av.size), av) if inward else
                                  (np.linspace(0, t_end, av.size), av[::-1])))
            ax.plot(av_t, e_true, "C0" + ls, label=f"true, {v:g} km/s")
        ax.plot(av_t, e_eq, "k-", lw=1.5, label="equilibrium (tables)")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(r"bulk $E_g$ of bin 4 [eV]", fontsize=13)
        ax.legend(frameon=False, fontsize=8.5)
        ax = axs[1, c]
        t_end, cond = profile_path(site, murga, 1.0, inward)
        tt = np.linspace(0, t_end, 300)
        av_t = np.interp(tt, np.linspace(0, t_end, av.size), av if inward else av[::-1])
        cs = [cond(x) for x in tt]
        tau = [bh.relaxation_time(bh.equilibrium(q), q) / YR for q in cs]
        ax.plot(av_t, [q.u_mathis for q in cs], "C1-", label=r"FUV [Mathis]")
        ax.plot(av_t, [q.n_hi for q in cs], "C2-", label=r"$n$(H\,{\sc i}) [cm$^{-3}$]")
        ax.plot(av_t, tau, "C3-", label="relaxation time [yr]")
        ax.plot(av_t, tt / YR if inward else (t_end - tt) / YR, "0.5", ls="--",
                label="time to the edge at 1 km/s [yr]")
        ax.set_yscale("log")
        ax.set_xlabel(r"$A_V$ [mag]", fontsize=13)
        ax.legend(frameon=False, fontsize=8.5)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--murga", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"),
                    help="directory holding Share_Acetylene/ (dataset murga2020-figshare)")
    ap.add_argument("--tables", default="results/carbon_plan/hydrogenated_tables.npz")
    ap.add_argument("--out", default="results/carbon_plan/check_hydrogen_tracking_T1.png")
    args = ap.parse_args(argv)

    grid = CarbonLayout().grid()
    hyd = {b: BinHydrogen(grid, b) for b in BINS}
    tabs = load_tables(args.tables)
    for b, bh in hyd.items():
        print(f"bin {b}: {grid.a_edges[b] * 1e7:.2f}-{grid.a_edges[b + 1] * 1e7:.2f} nm, H/C {bh.x_min:.3f} "
              f"(UV floor) to {bh.x_max:.3f} (H-addition plateau)")
    t1_maps(hyd, Path(args.out))

    print("\nT4/T6: largest error along the path against the true H/C: |dE_g| [eV]; |d ln kappa| at "
          + ", ".join(f"{p:g} um" for p in LAM_PROBE_UM))
    paths = []
    for site in ("orion_isobaric", "horsehead"):
        for v in SPEEDS_KMS:
            for inward in (True, False):
                t_end, cond = profile_path(site, args.murga, v, inward)
                paths.append((f"{site} {'in ' if inward else 'out'} {v:g} km/s ({t_end / YR:.2g} yr)", t_end, cond))
    for period in (1e4, 1e5, 1e6):
        t_end, cond = cycle_path(period)
        paths.append((f"CNM<->WNM, period {period:.0e} yr", t_end, cond))
    print("(outward paths start from the H plateau: hydrogenated material leaving the cloud)")
    for b in (4, 7):
        print(f"\n-- bin {b} --")
        for name, t_end, cond in paths:
            x0 = hyd[b].x_max if " out " in name else None
            res, (e_lo, e_hi), _ = run_path(hyd[b], tabs[b], t_end, cond, x0)
            print(f"{name:42s} true E_g {e_lo:.2f}-{e_hi:.2f}")
            for k, (de, dk) in res.items():
                print(f"    {k:28s} {de:6.3f} [{', '.join(f'{x:5.2f}' for x in dk)}]")
    plot_paths(hyd[4], tabs[4], args.murga, Path(args.out).with_name("check_hydrogen_tracking_paths.png"))

    print("\nT5: mixing two parcels 1:1 (equal carbon): kappa(mean H/C) / mean(kappa) - 1 at "
          + ", ".join(f"{p:g} um" for p in LAM_PROBE_UM))
    for b in (4, 7):
        bh = hyd[b]
        pairs = {"UV floor + H plateau": (bh.x_min, bh.x_max),
                 "CNM eq. + H plateau (a parcel from a cloud)": (bh.equilibrium(CNM), bh.x_max),
                 "CNM eq. + WNM eq.": (bh.equilibrium(CNM), bh.equilibrium(WNM))}
        for name, (x1, x2) in pairs.items():
            k1, k2 = np.exp(ln_kappa(tabs[b], bh.e_gap(x1))), np.exp(ln_kappa(tabs[b], bh.e_gap(x2)))
            km = np.exp(ln_kappa(tabs[b], bh.e_gap(0.5 * (x1 + x2))))
            bias = km / (0.5 * (k1 + k2)) - 1.0
            print(f"bin {b} {name:45s} " + "  ".join(f"{100 * x:+7.1f}%" for x in bias))


if __name__ == "__main__":
    main()
