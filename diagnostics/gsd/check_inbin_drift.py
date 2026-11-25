"""Size-dependent drift of one PL2M bin: does the in-bin state follow the grains as they segregate?

A bin's grains drift at w(a) relative to the gas (terminal velocity, here w proportional to a, as in the Epstein regime
under a size-independent acceleration). If the whole bin moves at one velocity, N and M move together and the in-bin
state s can never change: big and small grains of one bin cannot separate (McKinnon et al. 2018 note the same for
their dust particles). Diallo et al. (2026, RAMSES-MCR, Sect. 2.2.5, after Hopkins 2023) meet the same problem for
cosmic-ray diffusion in momentum bins and give the number and energy moments their own coefficients. For dust the
analogue is exact: N moves at w_N = int w dN / N and M at w_M = int w m dN / M, both from the bin's power law. The
packet carried across a face then has the mean mass int m w dN / int w dN, which lies inside the bin, so the scheme
stays realisable.

One grain bin (20-53 nm, eta = 18.8) in a periodic 1-D box, a top-hat of dust in the middle, first-order upwind for
every method:

* reference: the bin split into 64 sub-bins, each drifting at its own w (summed back into N, M);
* centre: N and M at the bin-centre velocity (s frozen);
* moments: N at w_N and M at w_M, recomputed from each cell's (N, M) every step (Diallo/Hopkins-type);
* two nodes: the collision operator's two nodes (median split), each at its own velocity;
* Gauss nodes: the 2-point Gauss rule of the bin in radius (exact for int a^k dN, k <= 3), the default rate nodes of
  the gsd methods since 2026-11-18.
The drift always uses the local power law of the bin (never EAF's isolated-bin pivot).

    python diagnostics/gsd/check_inbin_drift.py --out results/gsd_overview/figs/inbin_drift.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima.gsd.grid import MassGrid
from pycalima.gsd.pl2m import int_pow, mean_mass_ratio, solve_beta

RHO = 1.6
A_L, A_R = 19.96e-7, 53.1e-7
N_CELLS = 400
N_SUB = 64


def bin_grid():
    m = 4.0 / 3.0 * np.pi * RHO * np.array([A_L, A_R]) ** 3
    return MassGrid(m, RHO)


def moment_velocities(N, M, g, w0):
    """w_N, w_M per cell for w = w0 a/a_L, from each cell's power law."""
    eta = g.etas[0]
    filled = N > 0
    r = np.where(filled, M / np.where(filled, N, 1.0) / g.m_edges[0], 0.5 * (1 + eta))
    lam = 3.0 * np.atleast_1d(solve_beta(np.clip(r, 1.0 + 1e-12, eta * (1 - 1e-12)), eta)) + 2.0
    uR = eta ** (1.0 / 3.0)
    wN = w0 * int_pow(lam + 1, 1.0, uR) / int_pow(lam, 1.0, uR)
    wM = w0 * int_pow(lam + 4, 1.0, uR) / int_pow(lam + 3, 1.0, uR)
    return wN, wM


def two_nodes(N, M, g):
    """The two node masses of ``collisions.bin_nodes`` (number halves at the half means), per cell."""
    eta = g.etas[0]
    mL = g.m_edges[0]
    filled = N > 0
    r = np.clip(np.where(filled, M / np.where(filled, N, 1.0) / mL, 0.5 * (1 + eta)), 1.0 + 1e-12, eta * (1 - 1e-12))
    lam = 3.0 * np.atleast_1d(solve_beta(r, eta)) + 2.0
    uR = eta ** (1.0 / 3.0)
    p = lam + 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u_med = np.where(np.abs(p) < 1e-12, np.sqrt(uR), (0.5 * (1.0 + uR ** p)) ** (1.0 / p))
    m1 = mL * int_pow(lam + 3.0, 1.0, u_med) / int_pow(lam, 1.0, u_med)
    m1 = np.clip(np.nan_to_num(m1, nan=mL * r), mL, mL * eta)
    m2 = np.clip(2.0 * mL * r - m1, mL, mL * eta)
    return m1, m2


def gauss_nodes(N, M, g):
    """Masses and number fractions of the 2-point Gauss rule in radius of each cell's power law (exact for the
    integrals of u^k, k = 0-3, over dN, u = a/a_L)."""
    eta = g.etas[0]
    mL = g.m_edges[0]
    filled = N > 0
    r = np.clip(np.where(filled, M / np.where(filled, N, 1.0) / mL, 0.5 * (1 + eta)), 1.0 + 1e-12, eta * (1 - 1e-12))
    lam = 3.0 * np.atleast_1d(solve_beta(r, eta)) + 2.0
    uR = eta ** (1.0 / 3.0)
    i0 = int_pow(lam, 1.0, uR)
    m1, m2, m3 = (int_pow(lam + k, 1.0, uR) / i0 for k in (1, 2, 3))
    # nodes are the roots of x^2 + b x + c with m2 + b m1 + c = 0 and m3 + b m2 + c m1 = 0
    den = m2 - m1 * m1
    b = (m1 * m2 - m3) / den
    c = -m2 - b * m1
    disc = np.sqrt(np.maximum(b * b - 4.0 * c, 0.0))
    x1, x2 = (-b - disc) / 2.0, (-b + disc) / 2.0
    w1 = np.where(x2 > x1, (x2 - m1) / np.where(x2 > x1, x2 - x1, 1.0), 0.5)
    return (mL * x1 ** 3, mL * x2 ** 3), (w1, 1.0 - w1), (x1, x2)


def upwind(q, v, c):
    """One first-order upwind step on a periodic grid (v >= 0); c = dt/dx."""
    F = q * v
    return q - c * (F - np.roll(F, 1))


def run(t_end_frac=0.25, s0=0.138):
    g = bin_grid()
    eta = g.etas[0]
    beta0 = float(solve_beta(1.0 + s0 * (eta - 1.0), eta))
    lam0 = 3.0 * beta0 + 2.0
    w0 = 1.0                                     # velocity of the smallest grains; box length 1
    uR = eta ** (1.0 / 3.0)
    x = (np.arange(N_CELLS) + 0.5) / N_CELLS
    dx = 1.0 / N_CELLS
    shape = ((x > 0.1) & (x < 0.3)).astype(float)
    # reference sub-bins: edges in u, number and mass of the initial power law in each
    ue = np.geomspace(1.0, uR, N_SUB + 1)
    n_sub = np.array([int_pow(lam0, ue[k], ue[k + 1]) for k in range(N_SUB)])
    m_sub = g.m_edges[0] * np.array([int_pow(lam0 + 3, ue[k], ue[k + 1]) for k in range(N_SUB)])
    u_mid = np.array([int_pow(lam0 + 4, ue[k], ue[k + 1]) / int_pow(lam0 + 3, ue[k], ue[k + 1]) for k in range(N_SUB)])
    w_sub = w0 * u_mid
    Nref = shape[None, :] * n_sub[:, None]
    Mref = shape[None, :] * m_sub[:, None]
    N0, M0 = shape * n_sub.sum(), shape * m_sub.sum()
    w_max = w0 * uR
    t_end = t_end_frac / (w_max - w0)            # the fastest grains gain t_end_frac of the box on the slowest
    c = 0.5 * dx / w_max
    n_steps = int(np.ceil(t_end / (c * dx)))
    dt = t_end / n_steps
    cf = dt / dx
    a_c = g.a_centres[0]
    w_c = w0 * a_c / A_L
    Nc, Mc = N0.copy(), M0.copy()
    Nm, Mm = N0.copy(), M0.copy()
    Nt, Mt = N0.copy(), M0.copy()
    Ng, Mg = N0.copy(), M0.copy()
    for _ in range(n_steps):
        Nref = np.array([upwind(Nref[k], w_sub[k], cf) for k in range(N_SUB)])
        Mref = np.array([upwind(Mref[k], w_sub[k], cf) for k in range(N_SUB)])
        Nc, Mc = upwind(Nc, w_c, cf), upwind(Mc, w_c, cf)
        wN, wM = moment_velocities(Nm, Mm, g, w0)
        Nm, Mm = upwind(Nm, wN, cf), upwind(Mm, wM, cf)
        # two nodes per cell (collisions.bin_nodes, vectorised over cells)
        m1, m2 = two_nodes(Nt, Mt, g)
        w1, w2 = w0 * g.radius(m1) / A_L, w0 * g.radius(m2) / A_L
        FN = 0.5 * Nt * (w1 + w2)
        FM = 0.5 * Nt * (m1 * w1 + m2 * w2)
        Nt = Nt - cf * (FN - np.roll(FN, 1))
        Mt = Mt - cf * (FM - np.roll(FM, 1))
        (mg1, mg2), (wg1, wg2), (xg1, xg2) = gauss_nodes(Ng, Mg, g)
        FN = Ng * w0 * (wg1 * xg1 + wg2 * xg2)
        FM = Ng * w0 * (wg1 * mg1 * xg1 + wg2 * mg2 * xg2)
        Ng = Ng - cf * (FN - np.roll(FN, 1))
        Mg = Mg - cf * (FM - np.roll(FM, 1))
    ref = (Nref.sum(axis=0), Mref.sum(axis=0))
    return x, g, (N0, M0), ref, {"bin centre": (Nc, Mc), "moments (Diallo/Hopkins)": (Nm, Mm),
                                 "two nodes (median)": (Nt, Mt), "two nodes (Gauss)": (Ng, Mg)}


def s_of(N, M, g):
    eta = g.etas[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (M / N / g.m_edges[0] - 1.0) / (eta - 1.0)
    return np.where(N > 1e-6 * N.max(), s, np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/gsd_overview/figs/inbin_drift.pdf")
    args = ap.parse_args(argv)
    x, g, (N0, M0), (Nr, Mr), runs = run()
    lines = ["Drift of one grain bin (w ~ a), errors against 64 sub-bins: L1 of N and M (relative to the totals), "
             "and of s where the dust is (> 1e-3 of the peak number)."]
    sr = s_of(Nr, Mr, g)
    mask = Nr > 1e-3 * Nr.max()
    for k, (N, M) in runs.items():
        eN = np.abs(N - Nr).sum() / Nr.sum()
        eM = np.abs(M - Mr).sum() / Mr.sum()
        es = np.nanmean(np.abs(s_of(N, M, g)[mask] - sr[mask]))
        lines.append(f"  {k:26s} N {eN:.3f}  M {eM:.3f}  mean |ds| {es:.3f}")
    Path(args.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalima.plotting_style import use_calima_style
    use_calima_style()
    plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "legend.frameon": False})
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.4))
    styles = {"bin centre": ("C3", "--"), "moments (Diallo/Hopkins)": ("C0", "-"), "two nodes (median)": ("C2", ":"),
              "two nodes (Gauss)": ("C4", "-.")}
    for ax, (q0, qr, idx, lab) in zip(axs[:2], ((N0, Nr, 0, "number $N$"), (M0, Mr, 1, "mass $M$"))):
        ax.plot(x, q0 / q0.max(), c="0.7", lw=0.8, label="initial")
        ax.plot(x, qr / q0.max(), c="k", lw=1.6, label="reference (64 sub-bins)")
        for k, (c, ls) in styles.items():
            ax.plot(x, runs[k][idx] / q0.max(), c=c, ls=ls, lw=1.2, label=k)
        ax.set_xlabel("$x$ [box lengths]")
        ax.set_ylabel(lab + " (initial peak = 1)")
    axs[0].legend(fontsize=7, loc="upper right")
    ax = axs[2]
    ax.plot(x, sr, c="k", lw=1.6)
    for k, (c, ls) in styles.items():
        ax.plot(x, s_of(*runs[k], g), c=c, ls=ls, lw=1.2)
    ax.axhline(0.138, c="0.7", lw=0.8)
    ax.set_xlabel("$x$ [box lengths]")
    ax.set_ylabel("in-bin state $s$")
    ax.set_ylim(0, 0.8)
    for a in axs:
        a.set_xlim(0.15, 0.85)
    fig.suptitle(r"one grain bin (20--53 nm) drifting with $w \propto a$: the big grains run ahead", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
