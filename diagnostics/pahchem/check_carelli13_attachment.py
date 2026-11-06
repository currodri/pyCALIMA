#!/usr/bin/env python3
"""Electron attachment on neutral PAHs: what Carelli, Grassi & Gianturco (2013) say about pahchem's attachment fraction f.

Carelli+13 (A&A 549, A103, arXiv:1207.2673) compute the formation of temporary negative ions (TNI) from multichannel quantum scattering at
a fixed geometry (static exchange with model polarisation, Schwinger variational), all resonances and the threshold virtual state included,
for neutral anthracene, perylene and coronene (no dipole), and fit k_TNI(T) = alpha (T/300)^beta exp(-gamma/T) cm^3 s^-1 up to 1000 K
(their Table 1). They are an upper bound on the stable-anion rate, k_f <= k_TNI (autodetachment must not win), and they do not grow with
N_C: alpha = 1.31, 3.06, 2.74e-9 for C14, C20, C24.

pahchem's ``Params.attachment_fraction`` f is the share of classical captures that form the anion (it scales sigma_c). The classical
capture is ``chemistry.electron_capture(N_C, 0, T)`` (0.82 pi a^2 v J~, a = 0.9 sqrt(N_C) A), so the scattering result reads
f_TNI(T) = k_TNI(T) / k_capture(T). Then the end-to-end rate is k_capture f s_e(f), with s_e the statistical sticking of
``attachment.Attachment``, the low-density radiative limit, compared with the lab where there is one (anthracene, ~1e-9, collisionally
stabilised in a buffer). Nothing is tuned, and nothing here is a default.

Result (2026-11-01, 10-1000 K):
- f_TNI = 0.001-0.003 (anthracene), 0.004 (perylene), 0.003-0.004 (coronene): the ab initio rates (1-3e-9 cm^3 s^-1, flat in T
  and N_C) are 0.1-0.4 % of pahchem's classical capture (5-9e-7, growing as the geometric cross section). f_TNI N_C ~ 0.03 (C14),
  0.07-0.09 (C20, C24); at C54, if the TNI rate stays ~3e-9 (not computed above C24), f ~ 0.0015.
- Anthracene: Carelli's 1.31e-9 at 300 K matches the FALP 1.0e-9, which is collisionally stabilised: the scattering result and the
  lab agree on the capture step. pahchem's default (f = 1) attaches ~300x faster.
- End to end with f = f_TNI (radiative limit): coronene 1.7e-9 (10 K) to 2e-10 (1000 K), 0.07-0.82 of the TNI bound; anthracene
  3e-11 to 2e-10, 0.03-0.12 of the lab (the radiative-vs-collisional gap of V6, ``check_attachment_v6.py``).

    check_carelli13_attachment.py [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.pahchem import chemistry
from pycalima.pahchem.attachment import Attachment, grid_excess_cm
from pycalima.pahchem.molecule import PAHSpecies
from pycalima.plotting_style import use_calima_style

#: name -> (N_C, (alpha [cm^3 s^-1], beta, gamma [K])) from Carelli+13 Table 1
CARELLI = {"anthracene C14H10": (14, (1.31e-9, 0.21, -1.12)), "perylene C20H12": (20, (3.06e-9, -0.01, 2.77)),
           "coronene C24H12": (24, (2.74e-9, 0.11, -1.12))}
EA = {"anthracene C14H10": 0.53, "coronene C24H12": 0.47}                  # eV: check_attachment_v6.py; Andrews+16 Table A.1
T_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0)


def k_tni(par, t):
    a, b, g = par
    return a * (t / 300.0) ** b * np.exp(-g / t)


def anions():
    """{name: {label: (neutral, anion stand-in)}}: anthracene has no PAHdb anion (neutral's and cation's modes bracket it, as in V6)."""
    d = _paths.get_model_data_dir() / "PAH_states"
    ld = lambda f: PAHSpecies.from_states_file(d / f)                       # noqa: E731
    return {"anthracene C14H10": {"neutral modes": (ld("C14H10_0.dat"), ld("C14H10_0.dat")),
                                  "cation modes": (ld("C14H10_0.dat"), ld("C14H10_1.dat"))},
            "coronene C24H12": {"anion modes": (ld("C24H12_0.dat"), ld("C24H12_-1.dat"))}}


def sticking(neutral, anion, name, nc, t, f):
    e = grid_excess_cm(t)
    while True:
        at = Attachment(neutral, anion, EA[name], chemistry.disc_radius_cm(nc), e_excess_cm=e, capture_fraction=f)
        try:
            return at.sticking(t, np.array([0.0]), np.array([1.0]))
        except ValueError:
            e *= 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    print("f_TNI = k_TNI(Carelli+13) / k_capture(pahchem, sticking 1)")
    print(f"{'T [K]':>7s} " + " ".join(f"{n:>40s}" for n in CARELLI))
    print(f"{'':>7s} " + " ".join(f"{'k_TNI':>12s}{'k_capture':>14s}{'f_TNI':>12s}" for _ in CARELLI))
    f_t = {n: [] for n in CARELLI}
    for t in T_GRID:
        row = []
        for n, (nc, par) in CARELLI.items():
            kc, kt = chemistry.electron_capture(nc, 0, t), k_tni(par, t)
            f_t[n].append(kt / kc)
            row.append(f"{kt:12.2e}{kc:14.2e}{kt / kc:12.3f}")
        print(f"{t:7.0f} " + " ".join(row))
    print("\nf_TNI x N_C (a geometric capture grows as N_C; Carelli's rates do not):")
    for n, (nc, _) in CARELLI.items():
        print(f"  {n:20s} " + "  ".join(f"{t:.0f} K: {f * nc:5.2f}" for t, f in zip(T_GRID, f_t[n])))
    print("\nend-to-end k_att = k_capture f s_e(f), radiative limit (neutral in its ground state), f = f_TNI(T), against the TNI upper bound and the lab")
    lab = {"anthracene C14H10": {48: 1.7e-9, 75: 8.4e-10, 170: 2.6e-9, 300: 1.0e-9}}      # Moustefaoui+98, CRESU, normalised to FALP at 300 K
    sp = anions()
    for name, modes in sp.items():
        nc, par = CARELLI[name]
        for label, (neu, an) in modes.items():
            print(f"\n{name}, anion: {label}")
            print(f"{'T [K]':>7s} {'f_TNI':>8s} {'s_e(f_TNI)':>12s} {'k_att':>10s} {'k_att(f=1)':>11s} {'k_TNI':>10s} {'k_att/k_TNI':>12s} {'lab':>9s}")
            for t in T_GRID[:-1] if name.startswith("anthracene") else T_GRID:
                f = min(1.0, k_tni(par, t) / chemistry.electron_capture(nc, 0, t))
                kc = chemistry.electron_capture(nc, 0, t)
                s_f, s_1 = sticking(neu, an, name, nc, t, f), sticking(neu, an, name, nc, t, 1.0)
                k, k1, kt = kc * f * s_f, kc * s_1, k_tni(par, t)
                l = lab.get(name, {}).get(int(round(t)))
                print(f"{t:7.0f} {f:8.3f} {s_f:12.3e} {k:10.2e} {k1:11.2e} {kt:10.2e} {k / kt:12.2e} {('%.1e' % l) if l else '':>9s}")
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    use_calima_style()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    t = np.geomspace(10.0, 1000.0, 40)
    for n, (nc, par) in CARELLI.items():
        ax[0].loglog(t, k_tni(par, t), label=n.replace(" ", " (") + ")")
    for n, (nc, par) in CARELLI.items():
        ax[0].loglog(t, [chemistry.electron_capture(nc, 0, x) for x in t], ls=":", color="0.5")
        ax[1].semilogx(t, [k_tni(par, x) / chemistry.electron_capture(nc, 0, x) for x in t], label=n.split()[0])
    ax[0].set(xlabel=r"$T$ [K]", ylabel=r"$k$ [cm$^3$ s$^{-1}$]", title="TNI rate (solid), classical capture (dotted)")
    ax[1].set(xlabel=r"$T$ [K]", ylabel=r"$f_{\rm TNI}=k_{\rm TNI}/k_{\rm capture}$", ylim=(0, None))
    for a in ax:
        a.legend()
        a.minorticks_on()
        a.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out / "pahchem_carelli13_attachment.png", dpi=160)
    print(f"\nfigure in {out}")


if __name__ == "__main__":
    main()
