"""Optics tables for the EA closure at minimum memory: per-radius node tables instead of s tables.

With PL2M a bin's per-gram optics depend on the cell only through the in-bin state s, so RAMSES can tabulate
kappa_i(lambda, s) (step A2: 31-77 adaptive s points per bin for 3e-3). The edge-aware closures break this: EAF needs
four s tables per bin (base, two fronts, spike), and EA's fronts also depend on the neighbour's exponent, a second
table axis. All closures, however, are built from power-law pieces, and the per-gram average of a single-grain
quantity k(u) (u = a/a_L) over any piece is exact for k linear in u between nodes (``closures.bin_weights``). So the
alternative is to store k at a few radii per bin -- the per-radius node table -- and to integrate it over the cell's
own reconstruction at run time. Memory is then independent of the closure.

This script measures, for the A2 grids (carbon: THEMIS a-C(:H), silicate: astrosilicate Mie; 8 bins, 0.35/0.45 nm -
1 um; 61 wavelengths of the RAMSES grid; C_abs, C_sca, C_rp):

* node layouts per bin: uniform in u (A2's quadrature), geometric in u, and adaptive (bisection of ln u until the
  linear interpolation of k misses the midpoint value by less than tol at every wavelength and quantity);
* the error of kappa against a 512-node reference, on (i) PL2M profiles at 65 in-bin states per bin and (ii) EA and
  EAF states with fronts: power-law segments dn/da ~ a^q (q = -4.5 ... 1) with both cut-offs at many positions inside
  bins, plus random smooth states;
* memory (values per bin and wavelength) against the s tables, and the run-time cost of the weights.

    python diagnostics/gsd/check_node_tables.py --out results/gsd_overview/figs/node_tables.pdf

Needs $CALIMA_DUSTEM_DIR (THEMIS optics).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_optics_tables as A2   # noqa: E402

from pycalima.gsd import closures as gc   # noqa: E402
from pycalima.gsd.optics import node_table, node_u   # noqa: E402

LAM = A2.LAM_UM[::4]
N_REF = 512
N_FAMILY = (8, 12, 16, 24, 32, 48, 64, 96, 128)   # node counts of the uniform and geometric layouts
TOL_BEST = (1e-2, 3e-3, 1e-3)                     # per-bin minimal layouts that meet these on the bin averages
S_POINTS_A2 = {"carbon": {1e-2: (41, 36, 24, 22, 21, 20, 20, 26), 3e-3: (77, 60, 40, 35, 34, 31, 35, 46),
                          1e-3: (126, 104, 69, 55, 53, 55, 64, 76)},
               "silicate": {1e-2: (21, 21, 21, 21, 23, 25, 27, 26), 3e-3: (32, 32, 32, 33, 35, 41, 47, 46),
                            1e-3: (53, 53, 53, 56, 58, 72, 78, 79)}}   # results/binconv_v4/optics_tables_final.md


def adaptive_nodes(k_of_u, uR, tol, n_start=2, max_nodes=513):
    """Nodes in u (1 ... uR) bisected in ln u until linear interpolation of k(u) (all trailing entries) misses the
    value at each interval's midpoint (in ln u) by less than tol, relatively."""
    u = list(np.geomspace(1.0, uR, n_start + 1))
    cache = {x: np.asarray(k_of_u(np.array([x])))[0] for x in u}
    changed = True
    while changed and len(u) < max_nodes:
        changed = False
        new = [u[0]]
        for x0, x1 in zip(u[:-1], u[1:]):
            xm = np.sqrt(x0 * x1)
            if xm not in cache:
                cache[xm] = np.asarray(k_of_u(np.array([xm])))[0]
            t = (xm - x0) / (x1 - x0)
            lin = (1 - t) * cache[x0] + t * cache[x1]
            if np.max(np.abs(lin / cache[xm] - 1.0)) > tol:
                new.append(xm)
                changed = True
            new.append(x1)
        u = new
    return np.array(u), np.stack([cache[x] for x in u])


def layouts(grid, k, tols=(1e-2, 3e-3, 1e-3)):
    """{name: [(u, K) per bin]}: uniform and geometric node sets of several sizes, and adaptive ones."""
    out = {}
    uR = grid.etas ** (1.0 / 3.0)
    a_lo = grid.a_edges[:-1]
    for n in N_FAMILY:
        out[f"uniform {n}"] = [(node_u(e, n), np.asarray(k(a_lo[i] * node_u(e, n), LAM)))
                               for i, e in enumerate(grid.etas)]
        out[f"geometric {n}"] = [(np.geomspace(1.0, uR[i], n + 1),
                                  np.asarray(k(a_lo[i] * np.geomspace(1.0, uR[i], n + 1), LAM)))
                                 for i in range(grid.n_bins)]
    for tol in tols:
        out[f"adaptive {tol:g}"] = [adaptive_nodes(lambda u, i=i: k(a_lo[i] * u, LAM), uR[i], tol)
                                    for i in range(grid.n_bins)]
        tab = node_table(grid, lambda r: k(r, LAM), tol=tol)          # the production builder (gsd.optics)
        out[f"builder {tol:g}"] = list(zip(tab.u, tab.K))
    return out


def bin_kappa(P, i, u, K):
    """Per-gram kappa of bin i of the pieces P over nodes u with values K[len(u), ...]."""
    W = np.zeros((1, u.size))
    gc._piece_weights(np.ascontiguousarray(u[None, :]), P.lam[i:i + 1], P.u1[i:i + 1], P.u2[i:i + 1],
                      P.spike[i:i + 1], P.n[i:i + 1], 3.0, W)
    return np.tensordot(W[0], K, axes=(0, 0))


def segment(grid, lo, hi, q):
    """Exact (N, M) of dn/da = a^q on [lo, hi] over the bins of grid."""
    a = grid.a_edges
    x0, x1 = np.clip(a[:-1], lo, hi), np.clip(a[1:], lo, hi)
    rho = grid.rho
    return ((x1 ** (q + 1) - x0 ** (q + 1)) / (q + 1),
            4.0 / 3.0 * np.pi * rho * (x1 ** (q + 4) - x0 ** (q + 4)) / (q + 4))


def states(grid, rng, n_random=40):
    """(label, N, M) test states: front segments with cut-offs at many in-bin positions, and random smooth ones."""
    a = grid.a_edges
    out = []
    for q in (-4.5, -3.5, -2.0, 0.0, 1.0):
        for f_lo in (0.05, 0.3, 0.6, 0.9):
            for f_hi in (0.1, 0.5, 0.95):
                i_lo, i_hi = 1, grid.n_bins - 2
                lo = a[i_lo] * (a[i_lo + 1] / a[i_lo]) ** f_lo
                hi = a[i_hi] * (a[i_hi + 1] / a[i_hi]) ** f_hi
                N, M = segment(grid, lo, hi, q)
                out.append((f"segment q={q:g}", N, M))
        for lo_bin in range(grid.n_bins - 1):              # narrow segments: one or two bins, fronts on both sides
            lo = a[lo_bin] * (a[lo_bin + 1] / a[lo_bin]) ** 0.3
            hi = a[lo_bin + 1] * (a[lo_bin + 2] / a[lo_bin + 1]) ** 0.4
            out.append((f"narrow q={q:g}", *segment(grid, lo, hi, q)))
    for _ in range(n_random):
        N = rng.random(grid.n_bins) * 10.0 ** rng.uniform(-3, 0, grid.n_bins)
        s = rng.random(grid.n_bins)
        out.append(("random", N, N * grid.m_edges[:-1] * (1.0 + s * (grid.etas - 1.0))))
    return out


def evaluate(grid, k, rng):
    lay = layouts(grid, k)
    ref = [(node_u(e, N_REF), np.asarray(k(grid.a_edges[i] * node_u(e, N_REF), LAM))) for i, e in enumerate(grid.etas)]
    err = {name: {"pl2m s-sweep": 0.0, "ea fronts": 0.0, "eaf fronts": 0.0, "random": 0.0} for name in lay}
    err_bin = {name: np.zeros(grid.n_bins) for name in lay}       # worst over every state type, per bin
    # (i) PL2M profiles, 65 states per bin
    s = np.linspace(0.0, 1.0, 65)
    for i, e in enumerate(grid.etas):
        uref, Kref = ref[i]
        Wr = gc.state_weights(e, s, uref, *gc.component_pieces(e, s, "base", "pl2m"))
        kr = np.tensordot(Wr, Kref, axes=(1, 0))
        for name, L in lay.items():
            u, K = L[i]
            W = gc.state_weights(e, s, u, *gc.component_pieces(e, s, "base", "pl2m"))
            de = float(np.max(np.abs(np.tensordot(W, K, axes=(1, 0)) / kr - 1.0)))
            err[name]["pl2m s-sweep"] = max(err[name]["pl2m s-sweep"], de)
            err_bin[name][i] = max(err_bin[name][i], de)
    # (ii) EA and EAF states with fronts, and random states
    for label, N, M in states(grid, rng):
        for meth in ("ea", "eaf"):
            if label == "random" and meth == "eaf":
                continue
            P = gc.pieces(grid, N, M, meth)
            key = "random" if label == "random" else f"{meth} fronts"
            for i in np.nonzero(N > 1e-30 * N.max())[0]:
                kr = bin_kappa(P, i, *ref[i])
                for name, L in lay.items():
                    de = float(np.max(np.abs(bin_kappa(P, i, *L[i]) / kr - 1.0)))
                    err[name][key] = max(err[name][key], de)
                    err_bin[name][i] = max(err_bin[name][i], de)
    sizes = {name: [L[i][0].size for i in range(grid.n_bins)] for name, L in lay.items()}
    # the cheapest layout per bin that meets each tolerance on the bin averages (any family)
    best = {}
    for tol in TOL_BEST:
        pick = []
        for i in range(grid.n_bins):
            ok = [(sizes[name][i], name) for name in lay if err_bin[name][i] <= tol]
            pick.append(min(ok) if ok else (None, None))
        best[tol] = pick
    return err, sizes, lay, best


def cost(grid, lay_name, lay, n_rep=2000):
    """Run-time cost of the per-cell weights for one composition: time per call of the 8 bins' weights."""
    rng = np.random.default_rng(3)
    N, M = segment(grid, grid.a_edges[1] * 1.2, grid.a_edges[-2] * 1.5, -3.5)
    P = gc.pieces(grid, N, M, "ea")
    nodes = [L[0] for L in lay[lay_name]]
    nmax = max(u.size for u in nodes)
    for i in range(grid.n_bins):                            # warm the compiled kernel
        W = np.zeros((1, nodes[i].size))
        gc._piece_weights(np.ascontiguousarray(nodes[i][None, :]), P.lam[i:i + 1], P.u1[i:i + 1], P.u2[i:i + 1],
                          P.spike[i:i + 1], P.n[i:i + 1], 3.0, W)
    # one call over all bins with padded (repeated last) nodes, as a Fortran loop would do with its own counts
    U = np.stack([np.concatenate([u, np.full(nmax - u.size, u[-1] * (1 + 1e-12) ** np.arange(1, nmax - u.size + 1))])
                  for u in nodes])
    out = np.zeros(U.shape)
    gc._piece_weights(U, P.lam, P.u1, P.u2, P.spike, P.n, 3.0, out)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        gc._piece_weights(U, P.lam, P.u1, P.u2, P.spike, P.n, 3.0, out)
    return (time.perf_counter() - t0) / n_rep * 1e6, nmax, rng


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/gsd_overview/figs/node_tables.pdf")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(7)
    lines = ["Per-radius node tables for any closure (A2 grids, 61 wavelengths, C_abs/C_sca/C_rp): max relative error "
             "of the per-gram kappa against 512 uniform nodes, and nodes per bin.",
             "s tables of A2 for comparison (points per bin): " + "; ".join(
                 f"{m} tol {t:g}: {sum(v)} total" for m, d in S_POINTS_A2.items() for t, v in d.items())]
    results = {}
    for mat, grid, k in A2.materials():
        err, sizes, lay, best = evaluate(grid, k, rng)
        results[mat] = (err, sizes, best)
        lines.append(f"\n{mat}")
        lines.append(f"  {'layout':16s} {'nodes per bin':34s} {'total':>6s} {'PL2M s':>8s} {'EA fronts':>10s} "
                     f"{'EAF fronts':>10s} {'random':>8s}")
        for name in err:
            e = err[name]
            lines.append(f"  {name:16s} {' '.join(f'{x:3d}' for x in sizes[name]):34s} {sum(sizes[name]):6d} "
                         f"{e['pl2m s-sweep']:8.1e} {e['ea fronts']:10.1e} {e['eaf fronts']:10.1e} {e['random']:8.1e}")
        for tol, pick in best.items():
            n_ok = [n for n, _ in pick]
            if None in n_ok:
                lines.append(f"  best per bin for tol {tol:g}: not reached in every bin")
                continue
            lines.append(f"  best per bin for tol {tol:g} (on the bin averages): {' '.join(f'{n:3d}' for n in n_ok)}  "
                         f"total {sum(n_ok)}  [{', '.join(nm for _, nm in pick)}]")
        for name in ("adaptive 0.003", "uniform 64"):
            us, nmax, _ = cost(grid, name, lay)
            lines.append(f"  cost of the EA weights, {name}: {us:.1f} us per call for {grid.n_bins} bins "
                         f"(padded to {nmax} nodes; shared by every wavelength and quantity)")
    Path(args.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalima.plotting_style import use_calima_style
    use_calima_style()
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    marks = {"uniform": "o", "geometric": "s", "adaptive": "D", "builder": "P"}
    for ax, mat in zip(axs, ("carbon", "silicate")):
        err, sizes, best = results[mat]
        for name in err:
            fam = name.split()[0]
            tot = sum(sizes[name])
            worst_ea = max(err[name]["ea fronts"], err[name]["random"])
            ax.plot(tot, err[name]["pl2m s-sweep"], marks[fam], c="C1", mfc="none", ms=6)
            ax.plot(tot, worst_ea, marks[fam], c="C0", ms=5)
            if fam in ("adaptive", "builder"):
                ax.annotate(name.split()[1], (tot, worst_ea), textcoords="offset points", xytext=(4, 4), fontsize=7)
        for tol, v in S_POINTS_A2[mat].items():
            ax.plot(sum(v), tol, "*", c="k", ms=9)
            ax.annotate(f"s tables {tol:g}", (sum(v), tol), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("stored values per wavelength, all 8 bins")
        ax.set_title(f"{mat}: node tables (markers) vs s tables (stars)", fontsize=9)
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axs[0].set_ylabel(r"max relative error of $\kappa$")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], c="k", marker=m, ls="none", label=f"{f} nodes") for f, m in marks.items()]
    h += [Line2D([], [], c="C1", marker="o", mfc="none", ls="none", label="PL2M profiles (all s)"),
          Line2D([], [], c="C0", marker="o", ls="none", label="EA states (fronts, random)"),
          Line2D([], [], c="k", marker="*", ls="none", label="A2 s tables (PL2M only)")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
