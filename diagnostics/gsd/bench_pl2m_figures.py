"""Figures of the PL2M-variant benchmark (``bench_pl2m_variants.py figures``): reads the .pkl results in the output
directory and writes variants_*.pdf there."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_mckinnon18_tests as MK      # noqa: E402
import pl2m_variants as V                # noqa: E402

UM = MK.UM
# one colour per variant in every figure (the entity keeps its colour)
STYLE = {
    "pl2m": dict(c="k", ls="-", label="PL2M"),
    "tc:16": dict(c="#9ecae1", ls="-", label=r"TC, $L=16$"),
    "tc:8": dict(c="#4292c6", ls="-", label=r"TC, $L=8$"),
    "tc:4": dict(c="#08519c", ls="-", label=r"TC, $L=4$"),
    "ea:1e-2": dict(c="#d62728", ls="-", label=r"EA, $\epsilon=10^{-2}$"),
    "ea:1e-6": dict(c="#d62728", ls="--", label=r"EA, $\epsilon=10^{-6}$"),
    "ea:1e-2:8": dict(c="#d62728", ls=":", label=r"EA on TC ($L=8$)"),
    "aff": dict(c="#2ca02c", ls="-", label="affine + switch"),
    "exp": dict(c="#9467bd", ls="-", label=r"max-entropy, $e^{-\kappa m}$"),
    "PC": dict(c="#bbbbbb", ls="--", label="PC (McKinnon+18)"),
    "PL": dict(c="#8c6d31", ls="--", label="PL (McKinnon+18)"),
    # collision node variants
    "pl2m+gauss": dict(c="#ff7f0e", ls="-", label="Gauss nodes in $a$"),
    "pl2m+gaussm": dict(c="#ff7f0e", ls=":", label="Gauss nodes in $m$"),
    "pl2m+ka2": dict(c="#17becf", ls=":", label=r"kernel avg., $2\times2$"),
    "pl2m+ka4": dict(c="#17becf", ls="--", label=r"kernel avg., $4\times4$"),
    "pl2m+ka8": dict(c="#17becf", ls="-", label=r"kernel avg., $8\times8$"),
    "pl2m+gauss+ka4": dict(c="#8c564b", ls="-", label=r"Gauss + kernel avg. $4\times4$"),
}
A0_STYLE = {"pl2m": "pl2m", "tc8": "tc:8", "ea": "ea:1e-2", "ea_tc8": "ea:1e-2:8", "aff": "aff", "exp": "exp",
            "gauss": "pl2m+gauss", "ea_gauss": "ea_gauss", "ea6": "ea:1e-6", "ea6_gauss": "ea6_gauss",
            "eaf6": "eaf:1e-6:-3.5"}
STYLE["ea_gauss"] = dict(c="#e377c2", ls="--", label=r"EA ($\epsilon=10^{-2}$) + Gauss nodes")
STYLE["ea6_gauss"] = dict(c="#e377c2", ls="-", label=r"EA ($\epsilon=10^{-6}$) + Gauss nodes")
STYLE["eaf:1e-6:-3.5"] = dict(c="#bcbd22", ls="-", label=r"EAF (front exponent $-3.5$)")
STYLE["ea:1e-6"] = dict(c="#d62728", ls="-", label=r"EA, $\epsilon=10^{-6}$")
STYLE["ea:1e-2"] = dict(c="#d62728", ls="--", label=r"EA, $\epsilon=10^{-2}$")
MARK = {"pl2m": "o", "tc:16": "v", "tc:8": "v", "tc:4": "v", "ea:1e-2": "s", "ea:1e-6": "s", "ea:1e-2:8": "s",
        "aff": "^", "exp": "D", "PC": "x", "PL": "+", "pl2m+gauss": "P", "pl2m+gaussm": "P", "pl2m+ka2": "*",
        "pl2m+ka4": "*", "pl2m+ka8": "*", "pl2m+gauss+ka4": "X", "ea_gauss": "h", "ea6_gauss": "h",
        "eaf:1e-6:-3.5": "<"}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalima.plotting_style import use_calima_style
    use_calima_style(usetex=False)
    plt.rcParams.update({"mathtext.fontset": "cm", "font.serif": ["cmr10", "DejaVu Serif"],
                         "axes.formatter.use_mathtext": True, "legend.frameon": False,
                         "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})
    return plt


def ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", bottom=True, left=True, top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.6)


def load(out, name):
    p = Path(out) / f"{name}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def floor(x, lo=1e-14):
    return np.maximum(np.asarray(x, float), lo)


# ---------------------------------------------------------------------------------------------------------------------

def fig_growth(plt, out):
    g = load(out, "growth")
    st = load(out, "stress")
    if g is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axs[0]
    conv = g["conv_bins"]
    ns = np.array(conv["N"])
    for k in ["PC", "PL", "pl2m", "tc:8", "tc:4", "ea:1e-6", "aff", "exp"]:
        s = STYLE[k]
        ax.plot(ns, floor(conv[k]), marker=MARK[k], ms=4, c=s["c"], ls=s["ls"], lw=1.3, label=s["label"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 50)
    ax.set_xlabel(r"bins $N$ (1 nm to 1 $\mu$m)")
    ax.set_ylabel(r"$|M/M_{\rm exact}-1|$ at 5 Gyr")
    ax.set_title("(a) growth test vs bins, 100 steps", fontsize=10)
    ax.text(0.97, 0.2, r"EA: exact (round-off) for $N\geq64$", transform=ax.transAxes, ha="right", fontsize=8)
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    cs = g["conv_steps"]
    steps = np.array(cs["steps"])
    for ax, n, lab in ((axs[1], 8, "b"), (axs[2], 32, "c")):
        for k in ["PL", "pl2m", "tc:8", "tc:4", "ea:1e-6", "aff", "exp"]:
            s = STYLE[k]
            ax.plot(steps, floor(cs[(k, n)]), marker=MARK[k], ms=4, c=s["c"], ls=s["ls"], lw=1.3, label=s["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 2)
        ax.set_xlabel("steps over the 5 Gyr")
        ax.set_ylabel(r"$|M/M_{\rm exact}-1|$")
        ax.set_title(f"({lab}) growth test vs steps, $N={n}$", fontsize=10)
    axs[1].text(0.03, 0.03, "1 step: exact for every power-law\nclosure (edges on bin edges)", fontsize=8,
                transform=axs[1].transAxes)
    for ax in axs:
        ticks(ax)
    fig.tight_layout()
    fig.savefig(Path(out) / "variants_growth.pdf")
    plt.close(fig)


def fig_edges(plt, out):
    g = load(out, "growth")
    if g is None:
        return
    f9 = g["fig9"]
    specs = ["pl2m", "tc:8", "ea:1e-2", "aff", "exp"]
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.0), sharey="row")
    a, grid = f9["a"], f9["grid"]
    x = np.linspace(a[0], a[-1], 4000)
    for j, spec in enumerate(specs):
        ax = axs[0, j]
        cl = V.make_closure(spec)
        for k, col in ((0, "0.6"), (20, STYLE[spec]["c"])):
            sh = k * f9["da"]
            ax.fill_between(x / UM, ((x >= f9["lo"] + sh) & (x <= f9["hi"] + sh)).astype(float),
                            color="0.9" if k == 0 else "0.8", lw=0, zorder=0)
            N, M = f9[spec][k]
            xp, yp = V.dnda_curve(grid, N, M, cl, n_pts=300)
            ax.plot(xp / UM, yp, c=col, lw=1.4, label=("initial" if k == 0 else "after 20 steps"))
        for e in a:
            ax.axvline(e / UM, c="0.5", lw=0.5, ls=":")
        ax.set_ylim(-0.05, 2.6)
        ax.set_title(STYLE[spec]["label"], fontsize=10)
        ax.set_xlabel(r"$a$ [$\mu$m]")
    axs[0, 0].set_ylabel(r"$dn/da$ (boxcar = 1)")
    axs[0, 0].legend(fontsize=7.5, loc="upper right")
    axs[0, 0].text(0.03, 0.9, "(a) boxcar, 20 growth steps\n(exact: grey)", transform=axs[0, 0].transAxes,
                   fontsize=8, va="top")
    sin = g["sinusoid"][16]
    aa = sin["a"]
    from pycalima.gsd.grid import MassGrid
    g16 = MassGrid(MK.mass(aa), MK.RHO)
    xx = np.geomspace(aa[0], aa[-1], 4000)
    for j, spec in enumerate(["pl2m", "tc:8", "ea:1e-2", "aff", "PL"]):
        ax = axs[1, j]
        for ph, ls in ((0.5, "--"), (1.0, "-")):
            (st, (lo, hi)) = sin[spec]["snaps"][ph]
            ax.plot([MK.A_MIN / UM, lo / UM, lo / UM, hi / UM, hi / UM, MK.A_MAX / UM], [0, 0, 1, 1, 0, 0], c="0.6",
                    lw=0.9, ls=ls)
            if spec == "PL":
                b = MK.LinearBins(aa, st[0], constant=True)
                b.N, b.s = st
                ax.plot(xx / UM, b.dnda(xx), c=STYLE[spec]["c"], lw=1.3, ls=ls)
            else:
                xp, yp = V.dnda_curve(g16, st[0], st[1], V.make_closure(spec), n_pts=200)
                ax.plot(xp / UM, yp, c=STYLE[spec]["c"], lw=1.3, ls=ls)
        ax.set_xscale("log")
        ax.set_xlim(0.006, 0.2)
        ax.set_ylim(-0.05, 1.6)
        ax.set_xlabel(r"$a$ [$\mu$m]")
        ax.set_title(STYLE[spec]["label"] + r", $N=16$", fontsize=10)
    axs[1, 0].set_ylabel(r"$dn/da$")
    axs[1, 0].text(0.03, 0.95, "(b) growth-shrinking, McKinnon Fig. 13\ndashed: half period, solid: one period;\n"
                   "grey: exact", transform=axs[1, 0].transAxes, fontsize=7.5, va="top")
    for ax in axs.flat:
        ticks(ax)
    fig.tight_layout()
    fig.savefig(Path(out) / "variants_edges.pdf")
    plt.close(fig)


def fig_collisions(plt, out):
    c = load(out, "collisions")
    if c is None:
        return
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.2))
    names = [k[2] for k in c["mk"] if isinstance(k[1], int) and k[0] == "shat" and k[1] == 8]
    y = np.arange(len(names))
    for ax, kind, ns, title in ((axs[0, 0], "shat", (8, 16), "(a) shattering (McKinnon Fig. 15), 100 Myr"),
                                (axs[0, 1], "coag", (8, 32), "(b) coagulation (McKinnon Fig. 16), 20 Myr")):
        for jn, (n, mk) in enumerate(zip(ns, ("o", "s"))):
            for i, nm in enumerate(names):
                e = c["mk"].get((kind, n, nm))
                if e is None:
                    continue
                s = STYLE.get(nm, STYLE.get(nm.split("+")[0], dict(c="0.4")))
                ax.plot(e["errs"][-1]["misplaced"], i + 0.18 * (jn - 0.5), mk, c=s["c"], ms=6,
                        mfc=s["c"] if jn == 0 else "white")
                ax.plot(e["errs"][-1]["worstM"], i + 0.18 * (jn - 0.5), mk, c=s["c"], ms=3, alpha=0.4,
                        mfc="none")
        ax.set_yticks(y)
        ax.set_yticklabels([STYLE.get(nm, {"label": nm}).get("label", nm) if nm in STYLE else
                            STYLE[nm.split("+")[0]]["label"] if "+" not in nm else nm for nm in names], fontsize=8)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(r"mass misplaced $\sum|\Delta M_i|/M$ (small faint: worst bin)")
        ax.set_title(title + f"\nfilled circles: $N={ns[0]}$, open squares: $N={ns[1]}$", fontsize=9)
    # threshold sweeps
    for ax, kind, title in ((axs[0, 2], "shat", r"(c) shattering threshold swept, $N=8$"),
                            (axs[1, 0], "coag", r"(d) coagulation threshold swept, $N=16$")):
        keys = sorted({k[2] for k in c["sweep"] if k[0] == kind})
        for nm in keys:
            fs = sorted({k[1] for k in c["sweep"] if k[0] == kind and k[2] == nm})
            v = [c["sweep"][(kind, f, nm)]["misplaced"] for f in fs]
            s = STYLE.get(nm, dict(c="0.4", ls="-", label=nm))
            ax.plot(fs, v, marker=MARK.get(nm, "o"), ms=4, c=s["c"], ls=s["ls"], lw=1.2, label=s["label"])
        ax.set_yscale("log")
        ax.set_xlabel(r"threshold position $f=\ln(a_t/a_L)/\ln(a_R/a_L)$")
        ax.set_ylabel(r"$\sum|\Delta M_i|/M$")
        ax.set_title(title, fontsize=10)
        if kind == "coag":
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ * 3.0)
            ax.legend(fontsize=7, loc="upper right", ncol=2, title="(c) and (d)", title_fontsize=7)
    for ax, kernel, title in ((axs[1, 1], "const", r"(e) constant kernel, $\tau=100$"),
                              (axs[1, 2], "golovin", r"(f) additive (Golovin) kernel, $\tau=3$")):
        keys = sorted({k[2] for k in c["kernels"] if k[0] == kernel})
        for nm in keys:
            ns = sorted({k[1] for k in c["kernels"] if k[0] == kernel and k[2] == nm})
            v = [c["kernels"][(kernel, n, nm)][-1]["misplaced"] for n in ns]
            s = STYLE.get(nm, dict(c="0.4", ls="-", label=nm))
            ax.plot(ns, v, marker=MARK.get(nm, "o"), ms=4, c=s["c"], ls=s["ls"], lw=1.2, label=s["label"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("bins over 8 decades in mass")
        ax.set_ylabel(r"$\sum|\Delta M_i|/M$ vs analytic")
        ax.set_title(title, fontsize=10)
    h, l = axs[1, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", ncol=4, fontsize=7.5, bbox_to_anchor=(0.99, 0.0),
               title="(e) and (f)", title_fontsize=7.5)
    for ax in axs.flat:
        ticks(ax)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(Path(out) / "variants_collisions.pdf")
    plt.close(fig)


def stress_rows(st):
    """(label, {variant: error}) for the stress summary."""
    rows = []
    v = list(st["S1"][(8, 100, 1000)])
    rows.append((r"S1 narrow pop. $\times100$, $N=8$: $W_1$ (mass)", {k: st["S1"][(8, 100, 1000)][k]["w1M"] for k in v}))
    rows.append((r"S1 narrow pop. $\times10$, $N=16$: $|\Delta M/M|$", {k: st["S1"][(16, 10, 1000)][k]["merr"] for k in v}))
    for ic in ("boxcar", "mrn"):
        d = st["S2"][(16, ic)]
        rows.append((f"S2 10 growth-shrink periods, {ic}, $N=16$: L1(M)", {k: d[k]["l1M"][-1] for k in v}))
    rows.append((r"S3 equilibrium radius, $N=8$, $10\tau$: L1(M)", {k: st["S3"][(8, k)][-1]["l1M"] for k in v}))
    rows.append((r"S3 equilibrium radius, $N=16$, $3\tau$: L1(M)", {k: st["S3"][(16, k)][1]["l1M"] for k in v}))
    for sign, lab in ((1, "growth"), (-1, "shrinking")):
        rows.append((f"S4 edges, {lab}, $N=8$: max over f of $W_1$ (mass)",
                     {k: max(st["S4"][(8, sign, f)][k]["w1M"] for f in (0.05, 0.25, 0.5, 0.75, 0.95)) for k in v}))
    rows.append((r"S5 mixing (static), $N=8$: max $|\Delta S/S|$",
                 {k: max(abs(st["S5"][("static", 8, d)][k]["S"]) for d in (1e-7, 3e-7, 1e-6)) for k in v}))
    rows.append((r"S5 mixing (200 cycles), $N=8$: L1(M)", {k: st["S5"][("cycled", 8)][k]["l1M"] for k in v}))
    for gname in ("silicate8", "carbon"):
        rows.append((f"S8 {gname}, $10^5$ steps of $10^2$ yr: $W_1$ (mass)",
                     {k: st["S8"][(gname, 1e2)][k]["w1M"] for k in v}))
        rows.append((f"S8 {gname}, $10^5$ steps: $|\\Delta S/S|$ (surface)",
                     {k: abs(st["S8"][(gname, 1e2)][k]["S"]) for k in v}))
    return rows


def fig_stress(plt, out):
    st = load(out, "stress")
    if st is None:
        return
    rows = stress_rows(st)
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.6))
    vs = ["pl2m", "tc:8", "tc:4", "ea:1e-6", "aff", "exp"]
    for i, (lab, d) in enumerate(rows):
        for j, k in enumerate(vs):
            s = STYLE[k]
            ax.plot(max(d[k], 1e-7), i + 0.1 * (j - 2.5), MARK[k], c=s["c"], ms=6,
                    label=s["label"] if i == 0 else None, mfc=s["c"] if k != "pl2m" else "white")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 5)
    ax.set_xlabel(r"error (each row's metric; values below $10^{-7}$ drawn at $10^{-7}$)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0 + 1.0 / len(rows)), ncol=6)
    ticks(ax)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    fig.tight_layout()
    fig.savefig(Path(out) / "variants_stress.pdf")
    plt.close(fig)


def fig_a0(plt, out):
    a0 = load(out, "a0")
    rt = load(out, "rates")
    dr = load(out, "drift")
    if a0 is None:
        return
    cases = ("sn_diffuse", "mrn_accretion", "mrn_coagulation", "mrn_shattering", "mrn_sputtering")
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
    tags = ["stored", "pl2m", "tc8", "ea", "ea6", "eaf6", "aff", "exp", "gauss", "ea6_gauss"]
    for j, case in enumerate(cases):
        ax = axs.flat[j]
        for tag in tags:
            if tag == "stored":
                continue
            nbs = sorted(k[2] for k in a0 if k[0] == tag and k[1] == case)
            if not nbs:
                continue
            s = STYLE[A0_STYLE[tag]]
            ax.plot(nbs, [a0[(tag, case, n)]["err"] for n in nbs], marker=MARK[A0_STYLE[tag]], ms=4, c=s["c"],
                    ls=s["ls"], lw=1.2, label=s["label"])
        ax.axhline(0.05, c="0.6", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([4, 6, 8, 12])
        ax.set_xticklabels(["4", "6", "8", "12"])
        ax.set_xlabel("bins per composition")
        ax.set_title(case.replace("_", " "), fontsize=10)
        if j == 0:
            ax.set_ylabel("worst observable error")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[7].legend(h, l, fontsize=8, loc="center", title="variants (all panels)", title_fontsize=8)
    # rates at 8 bins: worst over cases, exact-integral and two-node
    ax = axs.flat[5]
    if rt is not None:
        tg = [t for t in tags[1:] if any(k[0] == t for k in rt)]
        xs = np.arange(len(tg))
        for nb, mk in ((8, "o"), (4, "s")):
            ex = [max([rt[k][0] for k in rt if k[0] == t and k[2] == nb] or [np.nan]) for t in tg]
            tw = [max([rt[k][1] for k in rt if k[0] == t and k[2] == nb] or [np.nan]) for t in tg]
            for i, t in enumerate(tg):
                s = STYLE[A0_STYLE[t]]
                ax.plot(i - 0.12, ex[i], mk, c=s["c"], ms=6 if nb == 8 else 4)
                ax.plot(i + 0.12, tw[i], mk, c=s["c"], mfc="white", ms=6 if nb == 8 else 4)
        ax.set_xticks(xs)
        ax.set_xticklabels([STYLE[A0_STYLE[t]]["label"] for t in tg], rotation=40, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel(r"rates: worst $|$run/ref$-1|$")
        ax.set_title("(rates) filled: exact in-bin integral,\nopen: two nodes; large $N=8$, small $N=4$",
                     fontsize=9)
    ax = axs.flat[6]
    if dr is not None:
        keys = [k for k in dr["runs"]]
        for i, k in enumerate(keys):
            eN, eM, es = dr[k]
            ax.bar(i - 0.2, eN, 0.4, color="0.55")
            ax.bar(i + 0.2, eM, 0.4, color="0.25")
        ax.bar([], [], color="0.55", label="L1 of $N$")
        ax.bar([], [], color="0.25", label="L1 of $M$")
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels([k.replace("moments, ", "mom. ") for k in keys], rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("error vs 64 sub-bins")
        ax.set_title("(drift) one bin, $w\\propto a$", fontsize=9)
        ax.legend(fontsize=7)
    axs.flat[7].axis("off")
    for ax in list(axs.flat)[:7]:
        ticks(ax)
    fig.tight_layout()
    fig.savefig(Path(out) / "variants_a0_rates_drift.pdf")
    plt.close(fig)


def fig_cost(plt, out):
    cost = load(out, "cost")
    g = load(out, "growth")
    a0 = load(out, "a0")
    st = load(out, "stress")
    if cost is None or g is None:
        return
    tm = cost["timings"]

    def c(k):
        return tm[k]["kernel"] + tm[k].get("remap_kernel", 0.0)

    c0 = c("pl2m")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    specs = [k for k in ["pl2m", "tc:8", "tc:4", "ea:1e-6", "eaf:1e-6:-3.5", "aff", "exp"] if k in tm]
    tag_of = {"pl2m": "pl2m", "tc:8": "tc8", "ea:1e-6": "ea6", "eaf:1e-6:-3.5": "eaf6", "aff": "aff", "exp": "exp"}
    off = {"pl2m": (6, 4), "tc:8": (-38, 6), "tc:4": (-36, -12), "ea:1e-6": (7, -3), "eaf:1e-6:-3.5": (7, 4),
           "aff": (-30, 8), "exp": (-58, -12)}
    short = {"pl2m": "PL2M", "tc:8": "TC8", "tc:4": "TC4", "ea:1e-6": "EA", "eaf:1e-6:-3.5": "EAF",
             "aff": "affine", "exp": "exp"}
    i8 = list(g["conv_bins"]["N"]).index(8)
    scan = g["ea_scan"]
    panels = [("(a) growth test, $N=8$, 100 steps",
               {k: (g["conv_bins"][k][i8] if k in g["conv_bins"] else scan[k][0]) for k in specs})]
    if a0 is not None:
        cases = ("sn_diffuse", "mrn_accretion", "mrn_coagulation", "mrn_shattering", "mrn_sputtering")
        panels.append(("(b) A0: worst observable, $N=8$, max over cases",
                       {k: max(a0[(tag_of[k], cc, 8)]["err"] for cc in cases) for k in specs
                        if k in tag_of and all((tag_of[k], cc, 8) in a0 for cc in cases)}))
    if st is not None:
        d8 = st["S8"][("silicate8", 1e2)]
        panels.append((r"(c) S8 silicate, $10^5$ steps: $W_1$ (mass)", {k: d8[k]["w1M"] for k in specs if k in d8}))
    for ax, (title, d) in zip(axs, panels):
        for k, v in d.items():
            s_ = STYLE[k]
            xk = c(k) / c0
            ax.plot(xk, max(v, 1e-14), MARK[k], c=s_["c"], ms=8, mfc=s_["c"] if k != "pl2m" else "white")
            ax.annotate(short[k], (xk, max(v, 1e-14)), xytext=off[k], textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 20)
        ax.margins(y=0.15)
        ax.set_xlabel("closure + remap kernel per cell-step / PL2M")
        ax.set_ylabel("error")
        ax.set_title(title, fontsize=10)
        ticks(ax)
    fig.tight_layout()
    fig.savefig(Path(out) / "variants_cost.pdf")
    plt.close(fig)


def main(out):
    plt = style()
    for f in (fig_growth, fig_edges, fig_collisions, fig_stress, fig_a0, fig_cost):
        try:
            f(plt, out)
            print("ok", f.__name__)
        except Exception as e:      # keep the other figures
            import traceback
            traceback.print_exc()
            print("FAILED", f.__name__, e)
