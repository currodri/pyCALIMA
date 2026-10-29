"""Our TD-LDA library through Lin, Yang & Li 2023's 2175 A bump analysis (MNRAS 525, 2380; 2026-10-28).

Their method: Octopus spectra of 30 compact PAHs, -1/0/+1, on a 0.3 A grid with 3 A spheres, T = 20, and a width of
0.2 eV per transition. Our spectra reproduce their numbers with exp(-0.2 t), i.e. a Lorentzian of half width
0.2 eV. Each spectrum per C is fitted over 0-25 um^-1 with a Drude profile for the pi* <- pi bump (peak x0, width
gamma) plus a Fano profile for the sigma band (their eqs. 5-7). The band strength is a1 pi / (2 gamma) (eq. 8).
Mixtures are equal-weight means per charge, then over -1, 0 and +1 (eqs. 1-4). Their numbers below were read off
their figures; Lin+25 (A&A 694, A84), same group and method, gives the neutrals' x0 in its Fig. 2.

1. Per species, for the five molecules we share (pyrene, coronene, ovalene, C54H18, C96H24) in all three charges,
   against their Fig. 8 widths and Lin+25's neutral positions.
2. Size trend (their Fig. 4 groups, all compact): our compact molecules (pyrene, coronene, ovalene, C54H18, C96H24,
   C150H30, C216H36), each averaged over its charges. The non-compact isomers are listed apart; their Drude fits
   are broad, and a gamma at the bound (5) means the profile does not describe them.
3. Charge state (their Table 1): per-charge means of our molecules, with the dications as well.
4. Positions: how far B3LYP sTD-DFT (PySCF def2-SVP + std2, ``--stda``) moves the bump band of each neutral against
   TD-LDA. The shift that best aligns the two spectra over the bump (both with a 0.15 eV Gaussian), and the move of
   the strongest maximum.
5. Grid: coronene 0 at their 0.3 A against our 0.25 A.

    PYTHONPATH=diagnostics/pahchem python diagnostics/pahchem/octopus/compare_lin.py --stda ~/Documents/RAMSES_dev/pahspec/stda_neutrals_2026-10-27/results
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diagnostics/pahchem/octopus"))
import build_library as B  # noqa: E402
import plot_dial3_library as L  # noqa: E402
import plot_progress as P  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

EV_UM = 1.2398420                   # E [eV] = 1.23984 x [um^-1]
X_MAX = 25.0                        # um^-1: their fitting range
E = np.round(np.arange(0.02, X_MAX * EV_UM + 0.02, 0.02), 10)
CHARGES = (-1, 0, 1)
#: Lin+23 Fig. 8, bump width gamma [um^-1] of anion, neutral, cation (read off the figure, +-0.03)
LIN23_GAMMA = {"pyrene": (1.63, 1.01, 1.15), "coronene": (0.57, 0.55, 0.55), "ovalene": (1.01, 1.00, 1.00),
               "C54H18": (0.80, 0.61, 0.71), "C96H24": (1.16, 1.01, 1.04)}
#: Lin+25 Fig. 2, neutral bump peak x0 [um^-1] (read off the figure, +-0.02)
LIN25_X0 = {"pyrene": 5.00, "coronene": 4.76, "ovalene": 4.48, "C54H18": 4.28, "C96H24": 4.11, "C150H30": 3.93}
#: Lin+23 Fig. 4, size groups (charge-averaged): (N_C min, N_C max, mean N_C, x0, gamma)
LIN23_GROUPS = ((16, 22, 19.4, 4.65, 1.59), (24, 38, 30.6, 4.53, 1.44), (40, 66, 50.0, 4.31, 1.44),
                (78, 96, 90.8, 4.18, 1.65))
#: Lin+23 Table 1, per charge over their 30 molecules: a1 [1e-10], x0, gamma
LIN23_TABLE1 = {-1: (21.25, 4.35, 1.56), 0: (21.11, 4.41, 1.56), 1: (20.15, 4.45, 1.52)}
MEGATRON = {"pyrene": {z: P.ROOT / f"malloci_pyrene_{z}" for z in CHARGES},
            "coronene": {-1: P.ROOT / "coronene_-1_h0.25", 0: P.ROOT / "coronene_0_h0.25_T20",
                         1: P.ROOT / "coronene_1_h0.25"},
            "ovalene": {z: P.ROOT / f"malloci_ovalene_{z}" for z in CHARGES}}
NC_SMALL = {"pyrene": 16, "coronene": 24, "ovalene": 32}
GRID_TEST = P.ROOT / "coronene_0"   # 0.3 A, T = 20
STDA_NAME = {"coronene": "coronene_0"}
#: pyrene's sTD-DFT ran on the laptop (results/pahchem/stda_splice_2026-10-27); its 423 K vapour band: 5.48 eV
STDA_PATH = {"pyrene": REPO / "results/pahchem/stda_splice_2026-10-27/pyrene_0.stddft.e10/tda.dat"}
COMPACT = ("pyrene", "coronene", "ovalene", "C54H18", "C96H24", "C150H30", "C216H36")


def drude(x, x0, g):
    return x ** 2 / ((x ** 2 - x0 ** 2) ** 2 + x ** 2 * g ** 2)


def fano(x, x0, g, q):
    return ((x - x0) + q * g / 2) ** 2 / ((x - x0) ** 2 + (g / 2) ** 2)


def model(x, a1, x01, g1, a2, x02, g2, q):
    return a1 * drude(x, x01, g1) + a2 * fano(x, x02, g2, q)


def fit(per_c_mb):
    """Their eq. 5 fitted to sigma / N_C [Mb] on E over 0-25 um^-1: (a1, x0_1, gamma_1, a2, x0_2, gamma_2, q),
    with a1 in their 1e-10 unit (= Mb um^-2)."""
    x = E / EV_UM
    p, _ = curve_fit(model, x, per_c_mb, p0=(20.0, 4.4, 1.0, 2.0, 12.2, 7.3, 3.3),
                     bounds=((0.0, 3.0, 0.05, 0.0, 8.0, 1.0, 0.0), (500.0, 6.5, 5.0, 20.0, 16.0, 20.0, 10.0)),
                     maxfev=20000)
    return p


def strength(p):
    """a1 pi / (2 gamma) [1e-13 cm/C] (their eq. 8)."""
    return p[0] * np.pi / (2 * p[2]) * 0.1


def lin_sigma(run):
    """sigma [cm^2] on E with their width: exp(-0.2 t), cut at T = 20."""
    return P.octopus_sigma(run, damp=lambda t, T: np.exp(-0.2 * t), energies=E)[1]


def states():
    """{(label, z): (N_C, sigma / N_C [Mb], run)}: megatron's pyrene, coronene, ovalene and every finished library
    state, with their width."""
    out = {}
    for label, runs in MEGATRON.items():
        for z, run in runs.items():
            out[(label, z)] = (NC_SMALL[label], lin_sigma(run) / NC_SMALL[label] / 1e-18, run)
    for label, nc, nh, z, *_, done in L.states(L.ROOT, 20.0 - 1e-6):
        if done:
            run = L.ROOT / f"{label}_{z:+d}"
            out[(label, z)] = (nc, lin_sigma(run) / nc / 1e-18, run)
    return out


def mixture(S, keys):
    """Their eqs. 1-4: equal-weight mean per charge over ``keys``, then over the charges present."""
    per_z = [np.mean([S[k][1] for k in keys if k[1] == z], axis=0) for z in CHARGES if any(k[1] == z for k in keys)]
    return np.mean(per_z, axis=0)


def sticks(path):
    rows = [r.split() for r in Path(path).read_text().splitlines()]
    return np.array([[float(r[1]), float(r[2])] for r in rows if len(r) >= 3 and r[0].isdigit()])


def bump_shift(run, tda):
    """(TD-LDA's bump band [eV], alignment shift, move of the strongest maximum [eV]) from TD-LDA to the sTD-DFT
    sticks, both with a 0.15 eV Gaussian. The bump band is TD-LDA's strongest maximum at 3.5-7 eV, and the window
    is +- 1 eV around it. The Drude x0 is not used: for pyrene it lies 1.2 eV above the band."""
    e = np.round(np.arange(2.0, 8.5, 0.005), 10)
    lda = P.octopus_sigma(run, damp=lambda t, T: np.exp(-0.5 * (0.15 * t) ** 2), energies=e)[1]
    st = sticks(tda)
    hyb = sum(f * np.exp(-0.5 * ((e - en) / 0.15) ** 2) for en, f in st)
    band = (e > 3.5) & (e < 7.0)
    c = e[band][np.argmax(lda[band])]
    win = (e > c - 1.0) & (e < c + 1.0)
    shifts = np.arange(-0.3, 1.2, 0.005)
    def score(d):
        h = np.interp(e[win] + d, e, hyb)
        return np.dot(lda[win], h) / np.sqrt(np.dot(lda[win], lda[win]) * np.dot(h, h))
    best = shifts[np.argmax([score(d) for d in shifts])]
    k = (e > c - 1.0) & (e < c + 1.6)
    return c, best, e[k][np.argmax(hyb[k])] - c


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stda", type=Path, default=None, help="directory of <state>_b3lyp_def2-svp.stddft/tda.dat")
    a = ap.parse_args()
    use_calima_style()
    S = states()
    F = {k: fit(v[1]) for k, v in S.items()}
    print("1. Per species, their processing: x0, gamma [um^-1], strength [1e-13 cm/C]; Lin+23 gamma, Lin+25 x0")
    for label in LIN23_GAMMA:
        for i, z in enumerate(CHARGES):
            p = F[(label, z)]
            ref = f"gamma {LIN23_GAMMA[label][i]:.2f}" + (f", x0 {LIN25_X0[label]:.2f}" if z == 0 else "")
            print(f"  {label:9s} {z:+d}  x0 {p[1]:.2f}  gamma {p[2]:.2f}  strength {strength(p):.2f}   Lin: {ref}")
    print("   all our other states (x0, gamma, strength):")
    for (label, z), p in F.items():
        if label not in LIN23_GAMMA:
            print(f"  {label:15s} {z:+d}  {p[1]:.2f}  {p[2]:.2f}  {strength(p):.2f}")
    print("2. Size trend: compact molecules, charges present among -1/0/+1 averaged; Lin+23 Fig. 4 groups")
    groups = []
    for m in COMPACT:
        keys = [k for k in S if k[0] == m and k[1] in CHARGES]
        p = fit(mixture(S, keys))
        groups.append((m, S[keys[0]][0], p))
        print(f"  ours {m:9s} N_C {S[keys[0]][0]:3d}  x0 {p[1]:.2f}  gamma {p[2]:.2f}  strength {strength(p):.2f}"
              f"  (charges {', '.join(f'{k[1]:+d}' for k in keys)})")
    for lo, hi, nc, x0, g in LIN23_GROUPS:
        print(f"  Lin  N_C {lo}-{hi:<3d}   x0 {x0:.2f}  gamma {g:.2f}  (mean N_C {nc:.0f})")
    flagged = [f"{k[0]} {k[1]:+d}" for k, p in F.items() if p[2] > 4.9]
    print("  Drude width at the bound (profile fails): " + (", ".join(flagged) or "none"))
    print("3. Charge state, over our molecules that have all of -1, 0, +1 (their Table 1: a1, x0, gamma):")
    full = sorted({k[0] for k in S if all((k[0], z) in S for z in CHARGES)})
    for z in (-1, 0, 1, 2):
        mols = [m for m in full if (m, z) in S]
        p = fit(np.mean([S[(m, z)][1] for m in mols], axis=0))
        ref = LIN23_TABLE1.get(z)
        print(f"  {z:+d}: a1 {p[0]:.2f}  x0 {p[1]:.2f}  gamma {p[2]:.2f}  strength {strength(p):.2f}  ({len(mols)} mol.)"
              + (f"   Lin: a1 {ref[0]:.2f}  x0 {ref[1]:.2f}  gamma {ref[2]:.2f}" if ref else ""))
    shifts = {}
    if a.stda:
        print("4. Bump band, B3LYP sTD-DFT against TD-LDA [eV]: TD-LDA band, alignment shift, strongest maximum; "
              "x0 + shift [um^-1]")
        for (label, z), (nc, _, run) in S.items():
            tda = STDA_PATH.get(label, a.stda / f"{STDA_NAME.get(label, f'{label}_+0')}_b3lyp_def2-svp.stddft"
                                / "tda.dat")
            if z != 0 or not tda.exists():
                continue
            c, d, dmax = bump_shift(run, tda)
            shifts[label] = (nc, F[(label, 0)][1], d)
            print(f"  {label:15s} {c:.2f}  {d:+.2f} {dmax:+.2f}   x0 {F[(label, 0)][1]:.2f} -> "
                  f"{F[(label, 0)][1] + d / EV_UM:.2f}")
    p30, p25 = fit(lin_sigma(GRID_TEST) / 24 / 1e-18), F[("coronene", 0)]
    print(f"5. Grid, coronene 0: 0.30 A x0 {p30[1]:.3f} gamma {p30[2]:.3f} strength {strength(p30):.3f}; "
          f"0.25 A x0 {p25[1]:.3f} gamma {p25[2]:.3f} strength {strength(p25):.3f}")
    fig_compare(S, F, groups, shifts, _paths.get_plots_dir("pahchem/octopus") / "lin23_compare.png")


def fig_compare(S, F, groups, shifts, path):
    """(a) bump width, ours against Lin+23 Fig. 8, per species and charge; (b) bump peak against N_C: Lin+23's size
    groups, Lin+25's neutrals, our groups and neutrals, and our neutrals with B3LYP positions."""
    col = dict(zip((-1, 0, 1, 2), P.COLORS))
    mk = {-1: "v", 0: "o", 1: "^"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": (1, 1.5)})
    for i, z in enumerate(CHARGES):
        ax1.plot([LIN23_GAMMA[m][i] for m in LIN23_GAMMA], [F[(m, z)][2] for m in LIN23_GAMMA], mk[z], color=col[z],
                 ms=7, label=f"charge {z:+d}")
    ax1.plot([0.3, 1.9], [0.3, 1.9], color="k", lw=0.8, ls="--")
    offset = {"pyrene": (-38, 6), "C96H24": (6, -11)}          # the two neutrals sit at the same point
    for m in LIN23_GAMMA:
        ax1.annotate(m, (LIN23_GAMMA[m][1], F[(m, 0)][2]), xytext=offset.get(m, (5, -10)), textcoords="offset points",
                     fontsize=7.5)
    ax1.set_xlabel(r"Lin+23 Fig. 8: $\gamma$ [$\mu$m$^{-1}$]")
    ax1.set_ylabel(r"ours, their 0.2 eV width: $\gamma$ [$\mu$m$^{-1}$]")
    ax1.set_xlim(0.3, 1.9)
    ax1.set_ylim(0.3, 1.9)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    P.style(ax1)
    ax2.axhline(4.6, color="0.5", ls="--", lw=0.8)
    ax2.axhline(4.42, color="0.5", ls=":", lw=0.8)
    ax2.text(58.5, 4.615, "Milky Way, 2175 Å", fontsize=8, color="0.35")
    ax2.text(58.5, 4.435, "JADES-GS-z6-0, 2263 Å", fontsize=8, color="0.35")
    ax2.plot([g[2] for g in LIN23_GROUPS], [g[3] for g in LIN23_GROUPS], "s", ms=9, mfc="none", color="k", mew=1.4,
             label="Lin+23 size groups (charges averaged)")
    ax2.plot([S[(m, 0)][0] for m in LIN25_X0], list(LIN25_X0.values()), "x", ms=7, color="k",
             label="Lin+25 neutrals")
    ax2.plot([g[1] for g in groups], [g[2][1] for g in groups], "s", ms=9, mfc="none", color=P.COLORS[0], mew=1.4,
             label="ours, compact molecules (charges averaged)")
    iso = sorted((S[k][0], F[k][1]) for k in S if k[1] == 0 and k[0] not in COMPACT and F[k][2] < 4.9)
    ax2.plot(*zip(*iso), "d", ms=5, mfc="none", color=P.COLORS[2], label="ours, non-compact isomers, neutral (TD-LDA)")
    if shifts:
        shifts = {m: v for m, v in shifts.items() if m in COMPACT}
        for nc, x0, d in shifts.values():
            ax2.annotate("", (nc, x0 + d / EV_UM), (nc, x0), arrowprops=dict(arrowstyle="->", color=P.COLORS[1], lw=1))
        ax2.plot([v[0] for v in shifts.values()], [v[1] + v[2] / EV_UM for v in shifts.values()], "o", ms=4,
                 color=P.COLORS[1], label="ours, neutrals with the B3LYP sTD-DFT bump shift")
    ax2.set_xscale("log")
    ax2.set_xticks([16, 24, 32, 54, 96, 150, 216])
    ax2.set_xticklabels(["16", "24", "32", "54", "96", "150", "216"])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel(r"$N_{\rm C}$")
    ax2.set_ylabel(r"bump peak $x_0$ [$\mu$m$^{-1}$]")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.set_ylim(3.65, 5.65)
    P.style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
