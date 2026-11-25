"""Per-gram optics tables of PL2M bins: how fine must the tables be?

Step A2 of the RAMSES GSD plan. With two moments per bin, the in-bin profile
changes from cell to cell, so a bin's optics per gram are tabulated against its
in-bin state s = (M/(N m_L) - 1)/(eta - 1) (``pycalima.gsd.optics``), on a
per-bin s grid refined by bisection, and interpolated by RAMSES as ln kappa
linear in s. This script measures, for the 8-bin grids of the study (carbon:
THEMIS a-C(:H), 0.35 nm - 1 um; silicate: astrosilicate Mie, 0.45 nm - 1 um):

1. nodes: how many nodes per bin the quadrature of k(u) = C/m needs;
2. s grid: for each refinement tolerance, the points per bin and the error of
   the interpolation on C_abs, C_sca, C_rp at every wavelength of the RAMSES
   grid (241 points, 1 nm - 1 mm), checked on dense s points, and on the
   observables (A_V, R_V, 2175 A bump, UV slope, A_lambda/A_V) of the final
   8-bin states of the bin-convergence study;
3. T grid: the error of the per-bin (T, s) tables of the emitted power per
   gram P(T) = 4 pi int kappa_abs B_lambda dlambda and of the Rosseland mean,
   interpolated in (s, ln T) on ln P, and the resulting T_dust error.

The target is 0.5% on every quantity. Silicate Mie is computed without the
geometric-optics fast path of ``MieTheory``, whose switch at |m| x = 1000 is a
jump in k(a) (e^1.5 in C_sca at 1 nm). The carbon optics need the DustEM THEMIS
dataset ($CALIMA_DUSTEM_DIR); the observables need the study's results
(``check_bin_convergence.py run``) in the same $CALIMA_RESULTS.

Usage
-----
    python diagnostics/gsd/check_optics_tables.py
    python diagnostics/gsd/check_optics_tables.py --no-plots
"""
from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np

from pycalima import _paths

LAM_UM = np.logspace(-3, 3, 241)          # the RAMSES optics grid
OBS_UM = (0.1, 0.12, 0.15, 0.18, 0.2175, 0.25, 0.3, 0.36, 0.44, 0.55, 0.7, 0.9, 1.25,
          1.65, 2.2)                      # the study's observables (check_bin_convergence)
N_BINS = 8
A_MAX_CM = 1e-4
NODES = (16, 32, 64, 128)
NODES_REF = 512
TOLS = (1e-2, 3e-3, 1e-3)
N_T = (25, 50, 100, 200)
T_RANGE = (3.0, 3000.0)                   # K
S_TEST = np.unique(np.concatenate([np.linspace(0.0, 1.0, 513), np.geomspace(1e-7, 1e-2, 60),
                                   1.0 - np.geomspace(1e-7, 1e-2, 60)]))
QUANTITIES = ("C_abs", "C_sca", "C_rp")

H_CGS, C_CGS, K_CGS = 6.62607015e-27, 2.99792458e10, 1.380649e-16


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def materials():
    """(name, grid, per-gram k(a_cm, lam_um) -> (na, nlam, 3))."""
    from pycalima.gsd.grid import MassGrid
    from pycalima.gsd.shattering import SILICATE
    from pycalima.models.dust_radiation import themis_optics as th
    from pycalima.models.tools.mie_theory import MieTheory

    def carbon(a_cm, lam):
        c = th.cross_sections(a_cm * 1e4, lam)
        m = 4.0 / 3.0 * np.pi * th.RHO_G_CM3 * a_cm ** 3
        return np.stack([c[q].T / m[:, None] for q in QUANTITIES], axis=-1)

    mie = MieTheory()
    with _quiet():
        mie.load_dielectric_constants(
            str(_paths.get_optical_props_path() / "draine_lee_1984" / "eps_suvSil"), "suvSil")

    def silicate(a_cm, lam):
        out = np.empty((a_cm.size, lam.size, 3))
        for i, a in enumerate(a_cm):
            for j, w in enumerate(lam):
                qa, qs, g = mie.compute_grain_properties(a * 1e4, w, "suvSil", use_fast_path=False)
                out[i, j] = qa, qs, qa + (1.0 - g) * qs
        m = 4.0 / 3.0 * np.pi * SILICATE.rho * a_cm ** 3
        return out * (np.pi * a_cm ** 2 / m)[:, None, None]

    def grid(a_min, rho):
        m = 4.0 / 3.0 * np.pi * rho * np.array([a_min, A_MAX_CM]) ** 3
        return MassGrid(np.geomspace(m[0], m[1], N_BINS + 1), rho)

    return [("carbon", grid(3.5e-8, th.RHO_G_CM3), carbon),
            ("silicate", grid(4.5e-8, SILICATE.rho), silicate)]


class BinOptics:
    """Node values k(u) of every bin, and kappa(s) = W(s) @ k from them."""

    def __init__(self, grid, k, n_nodes, lam):
        from pycalima.gsd.optics import node_u

        self.grid, self.u = grid, node_u(grid.eta, n_nodes)
        self.K = [np.asarray(k(grid.a_edges[i] * self.u, lam)) for i in range(grid.n_bins)]

    def kappa(self, i, s):
        from pycalima.gsd.optics import profile_weights

        return np.tensordot(profile_weights(self.grid.eta, s, self.u), self.K[i], axes=(1, 0))

    def refine(self, i, tol):
        from pycalima.gsd.optics import refine_s

        return refine_s(lambda s: self.kappa(i, s), tol)


def rel(a, b):
    return float(np.max(np.abs(a / b - 1.0)))


# ---------------------------------------------------------------------------
# 1-2: nodes and s grid
# ---------------------------------------------------------------------------

def check_nodes(mats):
    lam = LAM_UM[::4]
    s = np.linspace(0.0, 1.0, 65)
    print(f"\n## 1. Nodes per bin (max relative error vs {NODES_REF} nodes, 61 wavelengths)\n")
    print("| material | " + " | ".join(f"{n} nodes" for n in NODES) + " |")
    print("|---|" + "---|" * len(NODES))
    for name, grid, k in mats:
        ref = BinOptics(grid, k, NODES_REF, lam)
        ex = np.stack([ref.kappa(i, s) for i in range(grid.n_bins)])
        errs = []
        for n in NODES:
            b = BinOptics(grid, k, n, lam)
            errs.append(rel(np.stack([b.kappa(i, s) for i in range(grid.n_bins)]), ex))
        print(f"| {name} | " + " | ".join(f"{e:.1e}" for e in errs) + " |")


def check_s_grid(optics):
    from pycalima.gsd.optics import lookup

    print(f"\n## 2a. Adaptive s grids (ln kappa linear in s, 241 wavelengths, checked on "
          f"{S_TEST.size} s points)\n")
    print("| material | tol | points per bin | " + " | ".join(QUANTITIES) + " |")
    print("|---|---|---|" + "---|" * len(QUANTITIES))
    grids = {}
    for name, opt in optics.items():
        exact = [opt.kappa(i, S_TEST) for i in range(opt.grid.n_bins)]
        for tol in TOLS:
            tabs = [opt.refine(i, tol) for i in range(opt.grid.n_bins)]
            grids[name, tol] = tabs
            errs = [max(rel(lookup(s, v, S_TEST)[..., q], ex[..., q])
                        for (s, v), ex in zip(tabs, exact, strict=True)) for q in range(3)]
            n_s = " ".join(str(s.size) for s, _ in tabs)
            print(f"| {name} | {tol:g} | {n_s} | " + " | ".join(f"{e:.1e}" for e in errs) + " |")
    return grids


def observables(A):
    i = {lam: k for k, lam in enumerate(OBS_UM)}
    AV = A[..., i[0.55]]
    t = (1 / 0.2175 - 1 / 0.3) / (1 / 0.15 - 1 / 0.3)
    out = {"A_V": AV, "R_V": AV / (A[..., i[0.44]] - AV),
           "bump": A[..., i[0.2175]] / (A[..., i[0.3]] + t * (A[..., i[0.15]] - A[..., i[0.3]])),
           "UV_slope": A[..., i[0.1]] / A[..., i[0.15]]}
    out.update({f"curve({lam})": A[..., k] / AV for k, lam in enumerate(OBS_UM) if lam != 0.55})
    return out


def study_states():
    """Final (s, M) per bin of the 8-bin PL2M runs of check_bin_convergence."""
    d = _paths.get_results_dir() / "gsd_bin_convergence"
    out = {}
    for f in sorted(d.glob(f"*_pl2m_N{N_BINS:02d}_amin3.50A.npz")):
        z = np.load(f)
        st = {}
        for name, key in (("carbon", "graphite"), ("silicate", "silicate")):
            N, M, me = z[f"{key}:N"], z[f"{key}:M"], z[f"{key}:m_edges"]
            eta = me[1] / me[0]
            r = np.where(N > 0, M / np.where(N > 0, N, 1.0) / me[:-1], 0.5 * (1 + eta))
            st[name] = (np.clip((r - 1) / (eta - 1), 0, 1), M)
        out[f.name.split("_pl2m")[0]] = st
    return out


def check_observables(mats):
    """Observables at the study states: exact kappa(s) against the adaptive
    tables, both at the observables' wavelengths."""
    from pycalima.gsd.optics import lookup

    states = study_states()
    if not states:
        print("\n(no study results found: skipping the observables)")
        return
    lam = np.array(OBS_UM)
    obs = {name: BinOptics(grid, k, 64, lam) for name, grid, k in mats}
    print("\n## 2b. Observables of the final 8-bin study states (max relative error)\n")
    print("| case | " + " | ".join(f"tol {t:g}" for t in TOLS) + " | worst metric |")
    print("|---|" + "---|" * len(TOLS) + "---|")
    tabs = {(name, tol): [opt.refine(i, tol) for i in range(opt.grid.n_bins)]
            for name, opt in obs.items() for tol in TOLS}
    for case, st in states.items():
        errs, worst = [], ""
        for tol in TOLS:
            A_ex = A_in = 0.0
            for name, opt in obs.items():
                s, M = st[name]
                for i in range(opt.grid.n_bins):
                    ex = opt.kappa(i, s[i:i + 1])[0]
                    s_t, v_t = tabs[name, tol][i]
                    tb = lookup(s_t, v_t, s[i:i + 1])[0]
                    A_ex = A_ex + 1.086 * (ex[:, 0] + ex[:, 1]) * M[i]
                    A_in = A_in + 1.086 * (tb[:, 0] + tb[:, 1]) * M[i]
            o_ex, o_in = observables(A_ex), observables(A_in)
            e = {q: abs(o_in[q] / o_ex[q] - 1) for q in o_ex}
            worst = max(e, key=e.get)
            errs.append(e[worst])
        print(f"| {case} | " + " | ".join(f"{x:.1e}" for x in errs) + f" | {worst} |")


# ---------------------------------------------------------------------------
# 3: T grid
# ---------------------------------------------------------------------------

def planck_lambda(lam_cm, T):
    x = H_CGS * C_CGS / (lam_cm[None, :] * K_CGS * T[:, None])
    return 2 * H_CGS * C_CGS ** 2 / lam_cm[None, :] ** 5 / np.expm1(np.minimum(x, 700.0))


def emission(kabs, T):
    """P(T) per gram [erg/s/g] and the Rosseland mean, shape (s, T), from
    kabs[s, lam] (trapezoid rule in lambda)."""
    lam_cm = LAM_UM * 1e-4
    w = np.zeros(lam_cm.size)
    w[1:] += 0.5 * np.diff(lam_cm)
    w[:-1] += 0.5 * np.diff(lam_cm)
    x = np.minimum(H_CGS * C_CGS / (lam_cm[None, :] * K_CGS * T[:, None]), 700.0)
    B = planck_lambda(lam_cm, T)                                   # (T, lam)
    dB = B * (x / T[:, None]) / -np.expm1(-x)
    P = 4 * np.pi * kabs @ (B * w).T
    kR = (dB @ w)[None] / ((1.0 / kabs) @ (dB * w).T)
    return P, kR


def check_t_grid(optics, grids, tol):
    from pycalima.gsd.optics import lookup

    T_dense = np.geomspace(*T_RANGE, 1001)
    lT = np.log(T_dense)
    print(f"\n## 3. (T, s) emission tables on the tol = {tol:g} s grids: ln P linear in "
          f"(s, ln T), T in [{T_RANGE[0]:g}, {T_RANGE[1]:g}] K\n")
    print("| material | quantity | " + " | ".join(f"n_T={n}" for n in N_T) + " |")
    print("|---|---|" + "---|" * len(N_T))
    for name, opt in optics.items():
        res = {"P_emit": [0.0] * len(N_T), "kappa_R": [0.0] * len(N_T), "T_dust": [0.0] * len(N_T)}
        for i in range(opt.grid.n_bins):
            P_ex, R_ex = emission(opt.kappa(i, S_TEST)[..., 0], T_dense)
            s_t, v_t = grids[name, tol][i]
            for n_k, n in enumerate(N_T):
                T_tab = np.geomspace(*T_RANGE, n)
                lt = np.log(T_tab)
                P_t, R_t = emission(v_t[..., 0], T_tab)
                P_s, R_s = lookup(s_t, P_t, S_TEST), lookup(s_t, R_t, S_TEST)    # (s_test, T_tab)
                P_in = np.exp(np.array([np.interp(lT, lt, np.log(row)) for row in P_s]))
                R_in = np.exp(np.array([np.interp(lT, lt, np.log(row)) for row in R_s]))
                res["P_emit"][n_k] = max(res["P_emit"][n_k], rel(P_in, P_ex))
                res["kappa_R"][n_k] = max(res["kappa_R"][n_k], rel(R_in, R_ex))
                # T_dust: invert the interpolated table at the exact P of every (s, T)
                T_inv = np.exp(np.array([np.interp(np.log(pe), np.log(row), lt)
                                         for pe, row in zip(P_ex, P_s, strict=True)]))
                res["T_dust"][n_k] = max(res["T_dust"][n_k],
                                         float(np.max(np.abs(T_inv / T_dense - 1)[:, 1:-1])))
        for q, errs in res.items():
            print(f"| {name} | {q} | " + " | ".join(f"{e:.1e}" for e in errs) + " |")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def plot(optics, grids, tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    names = list(optics)
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 8.0), squeeze=False)
    for j, name in enumerate(names):
        opt = optics[name]
        for b, ax in zip((0, opt.grid.n_bins // 2), axes[:, j], strict=True):
            for s in (0.0, 0.01, 0.1, 0.5, 1.0):
                ax.loglog(LAM_UM, opt.kappa(b, np.array([s]))[0, :, 0], label=rf"$s={s:g}$")
            a_lo, a_hi = opt.grid.a_edges[b] * 1e7, opt.grid.a_edges[b + 1] * 1e7
            n_s = grids[name, tol][b][0].size
            ax.set_title(rf"{name}, bin {b + 1}: {a_lo:.2g}--{a_hi:.2g} nm, {n_s} $s$ points",
                         fontsize=13)
            ax.set_ylabel(r"$\kappa_\mathrm{abs}\ [\mathrm{cm^2\,g^{-1}}]$", fontsize=15)
            ax.minorticks_on()
            ax.tick_params(which="both", axis="both", direction="in")
        axes[-1, j].set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=15)
    axes[0, 0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    path = _paths.get_plots_dir() / "gsd_optics_tables.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-nodes", type=int, default=64, help="nodes per bin for sections 2-3")
    ap.add_argument("--tol", type=float, default=1e-3, help="s-grid tolerance for section 3")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    mats = materials()
    check_nodes(mats)
    optics = {name: BinOptics(grid, k, args.n_nodes, LAM_UM) for name, grid, k in mats}
    grids = check_s_grid(optics)
    check_observables(mats)
    check_t_grid(optics, grids, args.tol)
    if not args.no_plots:
        plot(optics, grids, args.tol)


if __name__ == "__main__":
    main()
