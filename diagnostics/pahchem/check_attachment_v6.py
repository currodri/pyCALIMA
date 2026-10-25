#!/usr/bin/env python3
"""Electron attachment on neutral PAHs: pahchem's statistical sticking against the lab (validation V6).

pahchem's attachment is ``chemistry.electron_capture(nc, 0, T)`` (BT94/DS87, 0.82 pi a^2 v_e J~(tau, 0), disc
radius 0.9 sqrt(N_C) A) times the statistical sticking s_e of ``attachment.Attachment``: autodetachment by
detailed balance against the anion's harmonic IR cascade, the low-density (radiative) limit. Nothing is tuned.

- **Anthracene** (EA 0.53 eV). Neutral: PAHdb UID 265 (model_data/PAH_states/C14H10_0.dat). PAHdb (v3.20 and
  v4.00) holds no anthracene anion, so its modes are a stand-in: the neutral's (UID 265) or the cation's (UID
  266, C14H10_1.dat), whose IR intensities are nearer an anion's; the two bracket the IR cascade. Lab:
  Moustefaoui+98 CRESU (47.7-300 K, normalised to the FALP 1.0e-9 at 300 K), Canosa+94 FALP, ~2e-9 at
  380-480 K (as quoted by Moustefaoui+98). The lab buffer (N2, 0.6-2.7e16 cm^-3) can stabilise the anion by
  collisions; the "lab, strong collisions" column replaces P_stab by max(P_stab, k_c / (k_c + k_d)) with
  k_c = 6.3e-10 cm^3 s^-1 (N2 Langevin) x 1e16 cm^-3, an upper bound for the lab.
- **C60** (EA 2.68 eV): PAHdb v4.00 UIDs 717 (neutral) and 723 (anion), both present. Lab: Viggiano+10,
  1.2e-6 (400 K) and 8e-7 (625 K), x3 absolute.
- **Tielens (2021) eqs. 8.101-8.103** (thermionic emission by detailed balance with sigma = pi e^2 a / eps,
  as recorded in calima-pdr/docs/literature/pah_gas/tielens2021_notes.md):
  k ~ 2.3e15 (N_C/50) (T_f/1000 K) exp[-12 (EA/eV)(1000 K/T_i)] s^-1, here with T_f the neutral's canonical
  T at E - EA and T_i either the anion's at E (T_a) or at E - EA/2 (T_e, Klots' finite heat bath); the notes
  do not define T_i and T_f, so this is a reading. Its sigma is the
  Coulomb-focused one of neutral -> cation emission, not the neutral's polarisation/image form pahchem uses,
  so only the order of magnitude is a check. s_e is "ground" (neutral at E_n = 0, pahchem's convention,
  ``population.neutral_sticking``) or "thermal" (neutral canonical at T, as in a lab cell).

Result (2026-10-24; s_e "ground" unless stated):
- Anthracene: k_att = 5.0e-11 (48 K), 1.4e-11 (300 K), 9e-12 (480 K) with the neutral's modes as the anion's,
  1.9e-10, 5.5e-11, 3.5e-11 with the cation's: 0.005-0.18 of the lab 0.84-2.6e-9, falling x4-5 from 48 to
  480 K where the lab is flat (a thermal neutral falls far faster: 4.5e-13 at 300 K). s_e ~ 1e-4 because
  k_d(EA + kT) ~ 1e4-1e5 s^-1 >> k_IR ~ 1-4 s^-1. Strong collisions in the lab's N2 buffer would give s_e -> 1,
  k = 5.2e-7, 200-600x above the lab: the lab lies between the radiative and the collisional limits. FAIL
  against the x3 lab scatter, low by x5-200, in the direction the low-density limit allows; the missing anion
  modes are a x4 uncertainty.
- C60: s_e = 1.000 (k_d < 1e-15 s^-1); k_att = 1.66e-6 (400 K) and 1.70e-6 (625 K), 1.39 and 2.13 x Viggiano+10.
  PASS (x3).
- Tielens (2021) eqs. 8.101-8.103 (EA for IP): pahchem's k_d is 0.03-0.09 of it for anthracene with T_i = T_e
  (4e-4-0.03 with T_i = T_a), and 1e-6-6e-4 (T_e) for C60, where E - EA << EA puts the Arrhenius form outside
  its range. Either k_d gives s_e << 1 for anthracene and s_e = 1 for C60: the verdicts do not depend on it.

    check_attachment_v6.py [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.pahchem import ames_suite, chemistry, dos
from pycalima.pahchem.attachment import Attachment, grid_excess_cm
from pycalima.pahchem.molecule import PAHSpecies
from pycalima.plotting_style import use_calima_style

EA = {"anthracene": 0.53, "C60": 2.68}
NC = {"anthracene": 14, "C60": 60}
#: (T [K], k [cm^3 s^-1]); Moustefaoui+98 Table 3 (CRESU, normalised to FALP at 300 K)
LAB_ANTHRACENE = ((47.7, 1.7e-9), (74.5, 8.4e-10), (169.7, 2.6e-9), (300.0, 1.0e-9))
LAB_ANTHRACENE_HOT = ((380.0, 2e-9), (480.0, 2e-9))      # "about 2e-9, flat" [sec]
LAB_C60 = ((400.0, 1.2e-6), (625.0, 8e-7))                # Viggiano+10, x3 absolute
K_COLL = 6.3e-10 * 1e16                                   # N2 + anion Langevin x n(N2) [s^-1]
T_PLOT = np.geomspace(30.0, 700.0, 18)


def species():
    """{name: (neutral, {label: anion stand-in or modes})}."""
    d = _paths.get_model_data_dir() / "PAH_states"
    neu, cat = (PAHSpecies.from_states_file(d / f"C14H10_{z}.dat") for z in (0, 1))
    recs = {r["uid"]: r for r in ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"))
            if r["uid"] in (717, 723)}
    c60 = {u: ames_suite.species_from_record(recs[u], f"PAHdb v4.00 UID {u}") for u in (717, 723)}
    return {"anthracene": (neu, {"neutral modes": neu, "cation modes": cat}),
            "C60": (c60[717], {"anion (UID 723)": c60[723]})}


def attachment(neutral, anion, name, temperature_k, thermal):
    """(Attachment, s_e) with the grid raised until the thermal weights fit."""
    e = grid_excess_cm(temperature_k)
    while True:
        at = Attachment(neutral, anion, EA[name], chemistry.disc_radius_cm(NC[name]), e_excess_cm=e)
        try:
            s = at.sticking(temperature_k) if thermal else at.sticking(temperature_k, np.array([0.0]), np.array([1.0]))
            return at, s
        except ValueError:
            e *= 2.0


def lab_collisional(at, temperature_k):
    """s_e with strong N2 collisions (one-step bound, module docstring), the neutral in its ground state."""
    p = at.p_stab.copy()
    i = slice(at.i_ea + 1, None)
    at.p_stab[i] = np.maximum(p[i], K_COLL / (K_COLL + at.k_detach[i]))
    try:
        return at.sticking(temperature_k, np.array([0.0]), np.array([1.0]))
    finally:
        at.p_stab = p


def tielens_ktie(nc, ea_ev, t_i, t_f):
    """Tielens (2021) eqs. 8.101-8.103 [s^-1], EA in place of IP."""
    return 2.3e15 * (nc / 50.0) * (t_f / 1000.0) * np.exp(-12.0 * ea_ev * 1000.0 / t_i)


def tielens_compare(at, neutral, anion, name, excess_ev=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8)):
    """[(E - EA [eV], k_d pahchem, k_tie(T_i = T_a(E)), k_tie(T_i = T_e), T_a, T_e, T_f)] on ``at``'s grid.

    Temperatures are canonical (``dos.canonical_temperature``): T_a of the anion at E, T_e of the anion at
    E - EA/2 (Klots' finite-heat-bath emission temperature), T_f of the neutral at E - EA.
    """
    out = []
    for x in excess_ev:
        e = at.ea_cm + x * dos.CM_PER_EV
        if e > at.energy_cm[-1]:
            continue
        i = int(round(e / at.grain))
        t_a, t_e = (float(dos.canonical_temperature(anion.freqs_cm, v)) for v in (e, e - 0.5 * at.ea_cm))
        t_f = float(dos.canonical_temperature(neutral.freqs_cm, x * dos.CM_PER_EV))
        out.append((x, at.k_detach[i], tielens_ktie(NC[name], EA[name], t_a, t_f),
                    tielens_ktie(NC[name], EA[name], t_e, t_f), t_a, t_e, t_f))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    sp = species()
    curves, tie = {}, {}
    print("V6: electron attachment k = electron_capture(N_C, 0, T) x s_e [cm^3 s^-1]")
    for name, (neu, anions) in sp.items():
        lab = dict(LAB_ANTHRACENE + LAB_ANTHRACENE_HOT if name == "anthracene" else LAB_C60)
        t_tab = sorted(set(lab) | ({48.0, 75.0, 170.0, 300.0, 400.0} if name == "anthracene" else {300.0}))
        for label, an in anions.items():
            print(f"\n{name} (EA {EA[name]} eV, a = {chemistry.disc_radius_cm(NC[name]) * 1e8:.2f} A), anion: {label}")
            print("     T   capture    s_e(ground)  k(ground)  s_e(thermal)  k(thermal)  k(lab, strong coll.)"
                  "   lab    k(ground)/lab  k(thermal)/lab")
            for T in t_tab:
                cap = chemistry.electron_capture(NC[name], 0, T)
                at, s0 = attachment(neu, an, name, T, False)
                sc = lab_collisional(at, T) if name == "anthracene" else np.nan
                _, st = attachment(neu, an, name, T, True)
                k_lab = lab.get(T)
                rl = f"{k_lab:.2e}   {cap * s0 / k_lab:11.3f}   {cap * st / k_lab:12.3f}" if k_lab else ""
                print(f"  {T:5.1f}  {cap:.2e}  {s0:11.3e}  {cap * s0:.2e}  {st:12.3e}  {cap * st:.2e}"
                      f"  {cap * sc:18.2e}    {rl}")
            curves[(name, label)] = [(T, *(chemistry.electron_capture(NC[name], 0, T)
                                          * np.array([attachment(neu, an, name, T, th)[1] for th in (False, True)])))
                                     for T in T_PLOT]
            at, _ = attachment(neu, an, name, 300.0, False)
            tie[(name, label)] = tielens_compare(at, neu, an, name)
    print("\nAutodetachment k_d(E) [s^-1]: pahchem (detailed balance) against Tielens (2021) eqs. 8.101-8.103")
    print("  (k_tie with T_i = T_a, the anion at E, or T_e, the anion at E - EA/2; T_f the neutral at E - EA)")
    print("  molecule, anion modes          E-EA [eV]   T_a [K]  T_e [K]  T_f [K]   k_d pahchem   k_tie(T_a)  "
          "k_tie(T_e)   k_d/k_tie(T_a)  k_d/k_tie(T_e)")
    for (name, label), rows in tie.items():
        for x, kd, ka, ke, ta, te, tf in rows:
            print(f"  {name + ', ' + label:30s} {x:8.2f}  {ta:8.0f} {te:8.0f} {tf:8.0f}   {kd:11.3e}   {ka:10.3e}  "
                  f"{ke:10.3e}   {kd / ka:14.3g}  {kd / ke:14.3g}")
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    plot(curves, tie, out / "pahchem_attachment_v6.png")


def plot(curves, tie, path):
    use_calima_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"neutral modes": "#2a78d6", "cation modes": "#eb6834", "anion (UID 723)": "#2a78d6"}
    for ax, name in zip(axes[:2], ("anthracene", "C60"), strict=True):
        T = np.array([c[0] for c in curves[(name, next(k for n, k in curves if n == name))]])
        ax.loglog(T, [chemistry.electron_capture(NC[name], 0, t) for t in T], "k:", lw=1.2, label=r"capture, $s_e = 1$")
        for (n, label), c in curves.items():
            if n != name:
                continue
            c = np.array(c)
            ax.loglog(c[:, 0], c[:, 1], color=colors[label], lw=1.8, label=f"pahchem, {label}, ground")
            ax.loglog(c[:, 0], c[:, 2], color=colors[label], lw=1.4, ls="--", label=f"pahchem, {label}, thermal")
        if name == "anthracene":
            lab = np.array(LAB_ANTHRACENE)
            ax.errorbar(lab[:, 0], lab[:, 1], yerr=0.3 * lab[:, 1], fmt="ks", ms=5, label="Moustefaoui+98, Canosa+94")
            hot = np.array(LAB_ANTHRACENE_HOT)
            ax.plot(hot[:, 0], hot[:, 1], "k^", ms=6, label=r"380--480 K (as quoted)")
        else:
            lab = np.array(LAB_C60)
            ax.errorbar(lab[:, 0], lab[:, 1], yerr=[lab[:, 1] * (1 - 1 / 3), lab[:, 1] * 2], fmt="ks", ms=5,
                        label=r"Viggiano+10 ($\times 3$)")
        ax.set_xlim(T_PLOT[0], T_PLOT[-1])
        ax.set_ylim(*((1e-14, 2e-6) if name == "anthracene" else (2e-7, 1e-5)))
        ax.set_title(rf"{name}, EA = {EA[name]} eV", fontsize=14)
        ax.set_xlabel(r"$T$ [K]", fontsize=14)
        ax.set_ylabel(r"$k_{\rm att}$ [cm$^3$ s$^{-1}$]", fontsize=14)
        ax.legend(frameon=False, fontsize=7)
    ax = axes[2]
    for (name, label), rows in tie.items():
        r = np.array(rows)
        c = "#eb6834" if label == "cation modes" else ("#2a78d6" if name == "anthracene" else "#3a9e5f")
        ax.semilogy(r[:, 0], r[:, 1], color=c, lw=1.8, marker="o", ms=3, label=f"pahchem, {name}, {label}")
        ax.semilogy(r[:, 0], r[:, 2], color=c, lw=1.2, ls=":", label=rf"Tielens 21, {name}, $T_i = T_a$")
        ax.semilogy(r[:, 0], r[:, 3], color=c, lw=1.4, ls="--", label=rf"Tielens 21, {name}, $T_i = T_e$")
    ax.set_xlabel(r"$E - {\rm EA}$ [eV]", fontsize=14)
    ax.set_ylabel(r"$k_d$ [s$^{-1}$]", fontsize=14)
    ax.set_title("autodetachment", fontsize=14)
    ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in")
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
