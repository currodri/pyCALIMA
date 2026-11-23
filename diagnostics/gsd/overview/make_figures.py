"""Figures for the GSD overview (gsd_overview.tex) that are new or redone.

Inputs, relative to the build directory results/gsd_overview/ (git-ignored), where the figures are written:
- data/paths.npz: extract_paths.py, run on megatron over the cap runs of check_pah_bins_hydrogen.py;
- ../carbon_plan/pah_bins_obs/deep_k8.txt: the production chain's E-against-truth report;
- ../carbon_plan/hydrogen_tables/deep_k8/*.npz: the production tables;
- ../binconv_v4/ and variants/a0_prod/: the A0 runs of check_bin_convergence.py.

    PYTHONPATH=src python diagnostics/gsd/overview/make_figures.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

from pycalima.plotting_style import use_calima_style  # noqa: E402

# the report's build directory (git-ignored): figures, data and the PDF live there, the sources here
HERE = Path(__file__).resolve().parents[3] / "results" / "gsd_overview"
FIGS = HERE / "figs"
YR = 3.15576e7
BIN_LABELS = (r"bin 1: $N_{\rm C}$ 20--42", r"bin 2: 42--89", r"bin 3: 89--189", r"bin 4: 189--400")
COLORS = ("#1f5fbf", "#e07b1a", "#2a9d5c", "#c0392b")


def style():
    use_calima_style()
    plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "xtick.bottom": True, "ytick.left": True,
                         "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
                         "ytick.minor.visible": True, "legend.frameon": False})


def ticks(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)


# -- PAH hydrogen: E against the truth along paths --------------------------------------------------------------------
def fig_pah_paths(d):
    run = "deep_k8"
    cols = ((1, "Orion Bar, inward at 1 km/s", "log"), (3, "Horsehead, inward at 1 km/s", "log"),
            (5, r"CNM $\leftrightarrow$ WNM, period $10^4$ yr", "lin"),
            (7, r"$G_0$ 0.1 $\leftrightarrow$ 1 at $n$(H) = 10, period $10^3$ yr", "lin"))
    rows = (("cond", None), ("heating", r"PE heating per C [erg s$^{-1}$]"),
            ("carbon", r"C loss per C atom [s$^{-1}$]"), ("h2", r"H$_2$ released per C [s$^{-1}$]"),
            ("r_ali", r"$r_{\rm ali}$ (3.4/3.3 proxy)"))
    rates = list(d["rates"])
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(11.0, 11.5), sharex="col")
    for j, (p, title, xs) in enumerate(cols):
        t = d[f"p{p}_t"]
        for i, (key, lab) in enumerate(rows):
            ax = axs[i, j]
            ticks(ax)
            if key == "cond":
                ax.step(t, d[f"p{p}_g0"], where="pre", color="k", lw=1.4, label=r"$G_0$ [Habing]")
                ax.step(t, d[f"p{p}_nh"], where="pre", color="0.55", lw=1.4, label=r"$n$(H) [cm$^{-3}$]")
                ax.set_yscale("log")
                ax.set_title(title, fontsize=11)
                if j == 0:
                    ax.legend(fontsize=9, loc="best")
                    ax.set_ylabel("conditions", fontsize=12)
            else:
                for b in range(4):
                    nc = float(d[f"bin{b}_nc"])
                    for kind, ls in (("truth", "-"), ("eq", "--")):
                        a = d[f"{run}_b{b}_p{p}_{kind}"]
                        if key == "r_ali":
                            y = a[:, rates.index("uv_ali")] / np.maximum(a[:, rates.index("uv_aro")], 1e-300)
                        else:
                            y = a[:, rates.index(key)] / nc
                        y = np.where(y > 0, y, np.nan)
                        ax.step(t, y, where="pre", color=COLORS[b], ls=ls, lw=1.3 if kind == "truth" else 1.0,
                                label=BIN_LABELS[b] if (kind == "truth" and i == 1 and j == 0) else None)
                ax.set_yscale("log")
                if key == "carbon":
                    ax.axhline(1 / (1e9 * YR), color="0.6", lw=0.8, ls=":")
                if key == "r_ali":
                    ax.set_ylim(1e-4, 3)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=12)
            if xs == "log":
                ax.set_xscale("log")
            if i == len(rows) - 1:
                ax.set_xlabel("time along the path [yr]", fontsize=12)
    axs[1, 0].legend(fontsize=9, loc="lower left")
    axs[2, 0].text(0.03, 0.06, "solid: time-dependent truth\ndashed: equilibrium table (E)\ndotted: lost in 1 Gyr",
                   transform=axs[2, 0].transAxes, fontsize=9)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    fig.savefig(FIGS / "pah_paths.pdf")
    plt.close(fig)


# -- PAH hydrogen: score summary --------------------------------------------------------------------------------------
OBS = ("carbon", "heating", "cooling", "h2", "h_uptake", "r_ali")
OBS_TEX = ("C loss", "heating", "cooling", r"H$_2$", "H uptake", r"$r_{\rm ali}$")


def parse_scores(txt):
    """{class: [bin, obs] worst path budget} and the same for the worst cell, from a check_pah_bins_hydrogen report."""
    classes = ("PDR paths", r"cycles, $10^4$--$10^5$ yr", r"cycles, $10^3$ yr")
    out = {c: np.full((4, len(OBS)), np.nan) for c in classes}
    for b, block in enumerate(txt.split("=== bin N_C ")[1:]):
        sec = block.split("observables, E against the truth")[1].split("worst over the paths")[0].splitlines()[2:]
        for line in sec:
            if not line.strip():
                continue
            label = line[:56].strip()
            vals = re.findall(r"(>99 |-  |\d+\.\d+)\s*/\s*(\d+\.\d+|-  ?)", line[56:])
            c = (classes[0] if any(s in label for s in ("NGC", "Orion", "Horsehead"))
                 else classes[2] if "1e+03" in label else classes[1])
            for k, (_, bud) in enumerate(vals):
                if bud.strip() != "-":
                    out[c][b, k] = np.nanmax([out[c][b, k], float(bud)])
    return out


def fig_pah_scores(txt):
    sc = parse_scores(txt)
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.3), sharey=True)
    norm = LogNorm(1e-3, 10)
    for ax, (c, m) in zip(axs, sc.items()):
        im = ax.imshow(np.clip(np.nan_to_num(m, nan=1e-3), 1e-3, 10), cmap="magma_r", norm=norm, aspect="auto")
        for b in range(4):
            for k in range(len(OBS)):
                v = m[b, k]
                s = "--" if not np.isfinite(v) else (f"{v:.2f}" if v < 10 else f"{v:.0f}")
                ax.text(k, b, s, ha="center", va="center", fontsize=9, color="w" if np.isfinite(v) and v > 0.5 else "k")
        ax.set_xticks(range(len(OBS)))
        ax.set_xticklabels(OBS_TEX, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(4))
        ax.set_yticklabels(BIN_LABELS, fontsize=10)
        ax.set_title(c, fontsize=11)
        ax.tick_params(which="both", length=0)
        ax.minorticks_off()
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01)
    cb.set_label(r"path-budget error $\max(E/T, T/E) - 1$", fontsize=10)
    fig.savefig(FIGS / "pah_scores.pdf", bbox_inches="tight")
    plt.close(fig)


# -- PAH hydrogen: the super-H cap ------------------------------------------------------------------------------------
def fig_pah_cap(d):
    runs = list(d["runs"])
    rates = list(d["rates"])
    xt = (r"A16, +4", r"deep, +8", r"deep, +12", r"deep, +24")
    paths = ((1, "Orion Bar in", "-", "o"), (3, "Horsehead in", "--", "s"), (5, r"CNM $\leftrightarrow$ WNM", ":", "^"))
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for p, name, ls, mk in paths:
        t = d[f"p{p}_t"]
        dt = np.diff(np.r_[0.0, t])
        for b in range(4):
            r_avg, cap, h2 = [], [], []
            for run in runs:
                a = d[f"{run}_b{b}_p{p}_truth"]
                r_avg.append((a[:, rates.index("uv_ali")] * dt).sum() / max((a[:, rates.index("uv_aro")] * dt).sum(),
                                                                            1e-300))
                cap.append(d[f"{run}_b{b}_p{p}_atcap"].max())
                h2.append((a[:, rates.index("h2")] * dt).sum() / dt.sum())
            h2 = np.array(h2) / h2[-1] if h2[-1] > 0 else np.full(4, np.nan)
            kw = dict(color=COLORS[b], ls=ls, marker=mk, ms=4, lw=1.2)
            axs[0].plot(range(4), r_avg, **kw)
            axs[1].plot(range(4), cap, **kw)
            axs[2].plot(range(4), h2, **kw)
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"path-averaged $r_{\rm ali}$", fontsize=12)
    axs[1].set_ylabel("largest share at the cap", fontsize=12)
    axs[2].set_ylabel(r"path-averaged H$_2$ / its value at +24", fontsize=12)
    axs[2].set_yscale("log")
    for ax in axs:
        ticks(ax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xt, fontsize=10)
        ax.minorticks_off()
        ax.tick_params(axis="y", which="both", direction="in")
    from matplotlib.lines import Line2D

    h_bins = [Line2D([], [], color=COLORS[b], lw=2) for b in range(4)]
    h_paths = [Line2D([], [], color="0.3", ls=ls, marker=mk, ms=4) for _, _, ls, mk in paths]
    fig.legend(h_bins + h_paths, list(BIN_LABELS) + [n for _, n, _, _ in paths], loc="lower center", ncol=7,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1))
    fig.savefig(FIGS / "pah_cap.pdf")
    plt.close(fig)


# -- the production tables --------------------------------------------------------------------------------------------
def fig_tables(tdir):
    names = ("20-42", "42-89", "89-189", "189-400")
    fig, axs = plt.subplots(4, 3, figsize=(10.5, 12.0), sharex=True, sharey=True)
    for b, nm in enumerate(names):
        z = np.load(tdir / f"bin_nc{nm}.npz")
        it, ix = list(np.round(z["log_t"], 6)).index(2.0), list(np.round(z["log_xe"], 6)).index(-4.0)
        g0, nh = z["log_g0"], z["log_nh"]
        nc, nh0 = float(z["nc_mean"]), float(z["nh0_mean"])
        sl = lambda k: z[k][:, :, it, ix]                          # noqa: E731  [g0, nh]
        X, Y = np.meshgrid(nh, g0)
        tau = np.where(sl("carbon") > 0, nc / np.maximum(sl("carbon"), 1e-300) / YR, 1e20)
        panels = ((np.clip(tau, 1e2, 1e12), LogNorm(1e2, 1e12), "viridis", r"carbon lifetime [yr]"),
                  (sl("heating") / nc, LogNorm(1e-29, 1e-19), "inferno", r"PE heating per C [erg s$^{-1}$]"),
                  (sl("n_h") / nh0, TwoSlopeNorm(1.0, 0.0, 1.25), "RdBu_r", r"$\langle N_{\rm H}\rangle / N_{\rm H,0}$"))
        for j, (v, norm, cmap, lab) in enumerate(panels):
            ax = axs[b, j]
            im = ax.pcolormesh(X, Y, v, norm=norm, cmap=cmap, shading="nearest", rasterized=True, linewidth=0,
                               edgecolors="face")
            ax.grid(False)
            if j == 2:
                cs = ax.contour(X, Y, sl("at_cap"), levels=[0.5], colors="k", linewidths=1.0)
                ax.clabel(cs, fmt={0.5: "at cap 0.5"}, fontsize=7)
            if j == 0:
                cs = ax.contour(X, Y, tau, levels=[1e6, 1e9], colors="w", linewidths=0.8)
                ax.clabel(cs, fmt={1e6: "1 Myr", 1e9: "1 Gyr"}, fontsize=7)
                ax.set_ylabel(BIN_LABELS[b] + "\n" + r"$\log G_0$ [Habing]", fontsize=11)
            if b == 3:
                ax.set_xlabel(r"$\log n$(H\,{\sc i}) [cm$^{-3}$]", fontsize=11)
            if b == 0:
                ax.set_title(lab, fontsize=11)
            ticks(ax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(r"production tables (deep chain, cap +8): slices at $T = 100$ K, $x_e = 10^{-4}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "tables.pdf")
    plt.close(fig)


def _curves(grid, N, M, method):
    """[(a, dN/da)] of every piece of a gsd method's reconstruction (spikes skipped)."""
    from pycalima.gsd import closures as gc
    N = np.asarray(N, float)
    P = gc.pieces(grid, N, np.asarray(M, float), method)
    out = []
    for i in range(grid.n_bins):
        for j in range(P.kp):
            if P.n[i, j] <= 1e-6 * N[i] or P.spike[i, j]:
                continue
            u = np.geomspace(P.u1[i, j], P.u2[i, j], 80)
            out.append((grid.a_edges[i] * u, P.A[i, j] * u ** P.lam[i, j] / grid.a_edges[i]))
    return out


def _profile(grid, i, N, M, a):
    """dN/da of bin i of (N, M) at radii a (zero outside the bin)."""
    from pycalima.gsd.pl2m import BinProfile
    prof = BinProfile(grid, N, M)
    u = a / grid.a_edges[i]
    inside = (a >= grid.a_edges[i]) & (a <= grid.a_edges[i + 1])
    return np.where(inside, prof.A[i] * u ** prof.lam[i] / grid.a_edges[i], 0.0)


def fig_pl2m():
    """The PL2M scheme: in-bin profiles against s, s against beta, a reconstruction and a remap."""
    from pycalima.gsd.collisions import bin_nodes
    from pycalima.gsd.grid import MassGrid
    from pycalima.gsd.pl2m import mean_mass_ratio
    from pycalima.gsd.remap import remap
    import matplotlib.ticker  # noqa: F401

    rho = 1.6
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6))
    # (a) profiles inside a grain bin at several s, with the two collision nodes
    eta = 18.832
    m_e = np.array([1.0, eta]) * 4.0 / 3.0 * np.pi * rho * (2.82e-7) ** 3
    g1 = MassGrid(m_e, rho)
    a = np.geomspace(g1.a_edges[0], g1.a_edges[1], 400)
    ax = axs[0, 0]
    for s, c in zip((0.03, 0.138, 0.3, 0.6, 0.95), ("C0", "C1", "C2", "C3", "C4")):
        M = m_e[0] * (1.0 + s * (eta - 1.0))
        f = _profile(g1, 0, np.array([1.0]), np.array([M]), a)
        y = a * f                                            # dN/dln a
        ax.plot(a * 1e7, y / y.max(), c=c, label=rf"$s = {s:g}$" + (" (MRN)" if s == 0.138 else ""))
        mm, _ = bin_nodes(g1, np.array([1.0]), np.array([M]))
        for mn in mm[0]:
            ax.axvline(g1.radius(mn) * 1e7, ymin=0, ymax=0.08, c=c, lw=2)
    ax.set_xscale("log")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$a$ [nm] (bin 2.82--7.50 nm, $\eta = 18.8$)")
    ax.set_ylabel(r"$dN/d\ln a$ (peak = 1)")
    ax.set_title(r"(a) power law inside a bin; ticks: the two collision nodes", fontsize=9)
    ax.legend(fontsize=7.5)
    ticks(ax)
    # (b) s against beta
    ax = axs[0, 1]
    beta = np.linspace(-12, 12, 481)
    for eta_b, c, lab in ((2.115, "C0", r"molecular bins, $\eta = 2.1$"), (18.832, "C3", r"grain bins, $\eta = 18.8$")):
        r = mean_mass_ratio(beta, eta_b)
        ax.plot(beta, (r - 1) / (eta_b - 1), c=c, label=lab)
        for b, mk, nm in ((-11.0 / 6.0, "o", "MRN"), (-1.0, "s", r"flat $dN/d\ln m$"), (-2.0, "^", r"flat $dM/d\ln m$")):
            rb = mean_mass_ratio(np.array([b]), eta_b)[0]
            ax.plot(b, (rb - 1) / (eta_b - 1), mk, c=c, ms=5, label=nm if eta_b > 3 else None)
    ax.set_xlabel(r"in-bin exponent $\beta$ ($dN/dm \propto m^\beta$)")
    ax.set_ylabel(r"in-bin state $s = (\bar m/m_L - 1)/(\eta - 1)$")
    ax.set_title(r"(b) $s$ is a monotone function of $\beta$", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ticks(ax)
    # (c) a reconstruction: the SN-dust case of A0 after 5 Myr, 8 bins against the reference
    ax = axs[1, 0]
    runs = HERE.parent / "binconv_v4/gsd_bin_convergence"
    runs_eaf = HERE / "variants/a0_prod/gsd_bin_convergence"
    for path, meth, c, lab, lw in ((runs / "sn_diffuse_pl2m_ref_amin3.50A.npz", "pl2m", "k",
                                    "reference (30 bins per mass decade)", 1.6),
                                   (runs_eaf / "sn_diffuse_ea_N08_amin3.50A.npz", "ea", "C0", "EA (default), 8 bins", 1.1),
                                   (runs / "sn_diffuse_pl2m_N08_amin3.50A.npz", "pl2m", "C1", "PL2M, 8 bins", 1.0)):
        z = np.load(path, allow_pickle=True)
        me, N, M = z["silicate:m_edges"], z["silicate:N"], z["silicate:M"]
        rho_s = me[0] / (4.0 / 3.0 * np.pi * 4.5e-8 ** 3)
        g = MassGrid(me, rho_s)
        first = True
        for ai, dnda in _curves(g, N, M, meth):
            ax.plot(ai * 1e7, g.mass(ai) * ai * dnda, c=c, lw=lw, label=lab if first else None)   # mass per ln a
            first = False
        if "N08" in path.name and meth == "pl2m":
            for e in g.a_edges:
                ax.axvline(e * 1e7, c="0.6", lw=0.4, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 1e3)
    ax.set_ylim(1e-33, 1e-27)
    ax.set_xlabel(r"$a$ [nm]")
    ax.set_ylabel(r"$m\,dN/d\ln a$ [g per H, arbitrary]")
    ax.set_title(r"(c) silicate after 5 Myr of SN-dust processing in the WNM (A0)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ticks(ax)
    # (d) a remap step: a distribution ending inside a bin, grown by a constant da
    ax = axs[1, 1]
    m_e4 = np.array([1.0, eta, eta ** 2, eta ** 3, eta ** 4]) * 4.0 / 3.0 * np.pi * rho * (1.06e-7) ** 3
    g4 = MassGrid(m_e4, rho)
    a_e = g4.a_edges
    a_cut = a_e[2] * (a_e[3] / a_e[2]) ** 0.45
    dnda0 = lambda x: np.where((x >= a_e[0]) & (x <= a_cut), (x / a_e[0]) ** -3.5, 0.0)   # noqa: E731
    N0 = np.zeros(4)
    M0 = np.zeros(4)
    for i in range(4):
        xs = np.geomspace(a_e[i], a_e[i + 1], 4001)
        ys = dnda0(xs)
        N0[i] = np.trapezoid(ys, xs)
        M0[i] = np.trapezoid(ys * g4.mass(xs), xs)
    da = 1.0e-7
    aa = np.geomspace(a_e[0], a_e[-1], 3000)
    ax.fill_between(aa * 1e7, dnda0(aa), 1e-12, color="0.85", label="before (exact)")
    ax.plot(aa * 1e7, dnda0(aa - da), c="k", ls="--", label=r"exact image, $a \to a + \Delta a$")
    for meth, c, lab in (("ea", "C0", "EA (default) after the remap"), ("pl2m", "C1", "PL2M after the remap")):
        N1, M1, *_ = remap(g4, N0, M0, lambda x: x + da, method=meth)
        first = True
        for ai, dnda in _curves(g4, N1, M1, meth):
            ax.plot(ai * 1e7, dnda, c=c, lw=1.2, label=lab if first else None)
            first = False
    for e in a_e:
        ax.axvline(e * 1e7, c="0.5", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(a_e[0] * 1e7, 30.0)
    ax.set_ylim(1e-4, 3.0)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$a$ [nm]")
    ax.set_ylabel(r"$dN/da$ [arbitrary]")
    ax.set_title(r"(d) growth of a distribution that ends inside a bin ($N$, $M$ exact for both)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ticks(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "pl2m_scheme.pdf")
    plt.close(fig)


def main():
    style()
    FIGS.mkdir(exist_ok=True)
    d = np.load(HERE / "data/paths.npz")
    fig_pah_paths(d)
    fig_pah_scores((HERE.parent / "carbon_plan/pah_bins_obs/deep_k8.txt").read_text())
    fig_pah_cap(d)
    fig_tables(HERE.parent / "carbon_plan/hydrogen_tables/deep_k8")
    fig_pl2m()
    print("wrote", sorted(p.name for p in FIGS.glob("*.pdf")))


if __name__ == "__main__":
    main()
