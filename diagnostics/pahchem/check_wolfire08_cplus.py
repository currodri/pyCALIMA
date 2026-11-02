#!/usr/bin/env python3
"""C+ neutralisation on PAHs: pahchem's untuned rates against Wolfire+08's calibrated model (validation V2-V3).

Wolfire+08 (ApJ 680, 384) fit Phi_PAH = 0.4 to C I/C II along diffuse sight lines, with one N_C = 35 PAH at
x_PAH = 6e-7 and their Table 2 rates (DS87 x Phi). pahchem has no Phi: its rates are DS87 capture with the
statistical sticking, mutual neutralisation at 0.58 x DS87 (Shuman+14) and 0.9 x Langevin (Canosa+95). Here the
v4.00 molecules with N_C 32-40 share x_PAH = 6e-7 at W08's CNM conditions (G0 = 1.7 Habing, n = 30 cm^-3,
T = 100 K, x(C+) = 1.4e-4, x(H+) = 3e-4, x_e = 4.5e-4), and alpha(C+) per H is compared with W08's three-state
balance at Phi = 0.4 (and 1).

alpha(C+) / W08 (Phi = 0.4), 2026-10-25: Jochims 0.91, statistical 0.73, valence 0.72 (f(-1) 0.15, 0.11, 0.11).

The figure (``pahchem_v2_w08_cplus.png``) repeats the comparison for n = 3-300 cm^-3 at W08's G0, T and
fractional abundances (x_e = x(C+) + x(H+)): alpha(C+) per H and f(Z), W08 at Phi = 0.4 and 1. Over n = 3-300,
pahchem / W08(Phi = 0.4) = 0.65-1.10 (valence), 0.65-1.11 (statistical), 0.69-1.28 (Jochims); pahchem has fewer
anions than W08 at Phi = 0.4 above n ~ 10 (f(-1) 0.11 against 0.22 at n = 30).

Rerun on 2026-11-02 with the current defaults: section 4 of the gas-coupling doc, where ``check_default_changes.py``
adds the changes since 2026-10-25 one at a time.

    check_wolfire08_cplus.py [--out <plot dir>]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_34um_sidegroups as S

from pycalima import _paths
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

G0, N, T = 1.7, 30.0, 100.0
XC, XH = 1.4e-4, 3e-4


#: W08 Table 2 at T = 100 K, Phi = 1 [cm^3 s^-1]: (X+ + PAH-, X+ + PAH0)
W08_ION = {"C+": (2.4e-7, 8.8e-9), "H+": (8.3e-7, 3.1e-8)}


DETACHMENTS = {"valence": ("#2a78d6", "-"), "statistical": ("#1baf7a", "--"), "jochims": ("#eb6834", ":")}
N_GRID = np.geomspace(3.0, 300.0, 9)


def wolfire08(phi, x=6e-7, ion="C+", n=N):
    """(alpha(ion) per H [cm^3 s^-1], (f-, f0, f+)) of W08's Table 2 rates in a three-state balance."""
    ne, t2 = (XC + XH) * n, T / 100.0
    att, det = 1.3e-6 * phi * ne, 1.2e-8 * G0
    mn = phi * t2**-0.5 * (W08_ION["C+"][0] * XC * n + W08_ION["H+"][0] * XH * n)
    up = 4.6e-9 * G0 + phi * (W08_ION["C+"][1] * XC * n + W08_ION["H+"][1] * XH * n)
    rm, rp = att / (det + mn), up / (3.5e-5 * phi * t2**-0.5 * ne)
    f0 = 1.0 / (1.0 + rm + rp)
    k_mn, k_ct = W08_ION[ion]
    return x * phi * (k_mn * t2**-0.5 * rm * f0 + k_ct * f0), (rm * f0, f0, rp * f0)


def cell(n):
    return Environment("cnm", G0, n, (XC + XH) * n, T, None, ions=(("C+", XC * n), ("H+", XH * n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    for phi in (0.4, 1.0):
        a, f = wolfire08(phi)
        print(f"Wolfire+08 Phi = {phi}: alpha(C+) {a:.3e} cm^3 s^-1 per H; f(-, 0, +) {np.round(f, 3)}")
    lib = [m for m in S.library() if 32 <= m.nc <= 40 and -1 in m.charges]
    x = np.full(len(lib), 6e-7 / len(lib))
    sweep = {}
    for det in DETACHMENTS:
        pop = P.Population(lib, {"all": (1.0, 1e4)}, params=Params(detachment=det))
        pop.prepare(P.field_shape(cell(N)), workers=8)
        r = pop.gas_rates(cell(N), x)
        a = r["alpha_mn"]["C+"] + r["alpha_ct"]["C+"]
        fz = x @ r["f"] / x.sum()
        print(f"pahchem, {len(lib)} molecules (N_C 32-40), detachment {det}: alpha(C+) {a:.3e} (mutual "
              f"neutralisation {r['alpha_mn']['C+']:.2e}, ion-neutral {r['alpha_ct']['C+']:.2e}); "
              f"f(-, 0, +, ...) {np.round(fz, 3)}; / W08(Phi = 0.4) = {a / wolfire08(0.4)[0]:.2f}")
        rs = [pop.gas_rates(cell(n), x) for n in N_GRID]
        sweep[det] = (np.array([q["alpha_mn"]["C+"] + q["alpha_ct"]["C+"] for q in rs]),
                      np.array([x @ q["f"] / x.sum() for q in rs]), pop.zs)
    w = np.array([wolfire08(0.4, n=n)[0] for n in N_GRID])
    print(f"n = {N_GRID[0]:g}-{N_GRID[-1]:g}: alpha(C+) / W08(Phi = 0.4) "
          + ", ".join(f"{det} {(sweep[det][0] / w).min():.2f}-{(sweep[det][0] / w).max():.2f}" for det in DETACHMENTS))
    plot(sweep, (args.out or _paths.get_plots_dir("pahchem")) / "pahchem_v2_w08_cplus.png")


def plot(sweep, path):
    use_calima_style()
    fig, (ax, fx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.28})
    for phi, ls in ((0.4, "-"), (1.0, "--")):
        w = [wolfire08(phi, n=n) for n in N_GRID]
        ax.loglog(N_GRID, [q[0] for q in w], "k", ls=ls, lw=2, label=rf"Wolfire+08, $\Phi_{{\rm PAH}} = {phi:g}$")
        if phi == 0.4:
            for k, c in enumerate(("#2a78d6", "#eb6834", "#1baf7a")):
                fx.semilogx(N_GRID, [q[1][k] for q in w], color=c, ls="-", lw=1.2, alpha=0.6)
    for det, (c, ls) in DETACHMENTS.items():
        a, f, zs = sweep[det]
        lab = f"pahchem, {det}" + (" (default)" if det == Params().detachment else "")
        ax.loglog(N_GRID, a, color=c, ls=ls, lw=2.2, label=lab)
    a, f, zs = sweep[Params().detachment]
    for c, z, name in (("#2a78d6", -1, "PAH$^-$"), ("#eb6834", 0, "PAH$^0$"), ("#1baf7a", 1, "PAH$^+$")):
        fx.semilogx(N_GRID, f[:, zs.index(z)], color=c, lw=2.4, label=name)
    fx.plot([], [], color="0.4", lw=1.2, alpha=0.6, label=r"Wolfire+08, $\Phi = 0.4$")
    for a_ in (ax, fx):
        a_.axvline(N, color="0.7", lw=1, zorder=0)
        a_.set_xlabel(r"$n$ [cm$^{-3}$] ($x_e = 4.5\times10^{-4}$, $G_0 = 1.7$, $T = 100$ K)", fontsize=13)
        a_.minorticks_on()
        a_.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_ylabel(r"$\alpha({\rm C^+})$ on PAHs [cm$^3$ s$^{-1}$ H$^{-1}$]", fontsize=14)
    ax.set_title(r"C$^+$ neutralisation, $x_{\rm PAH} = 6\times10^{-7}$ ($N_{\rm C}$ 32--40)", fontsize=12)
    fx.set_ylabel(r"$f(Z)$", fontsize=14)
    fx.set_title(f"charge fractions, pahchem {Params().detachment} (thick)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fx.legend(frameon=False, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
