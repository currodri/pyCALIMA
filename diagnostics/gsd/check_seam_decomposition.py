#!/usr/bin/env python3
"""Decomposition of the PAH/grain seam: which ingredient makes the PAH tables heat the gas more per C than 'ac'?

Seam study of 2026-11-23 (results/carbon_plan/carbon_transfer_2026-11-23/), step 1. It redoes, with today's pahchem
defaults, the October ingredient swap whose script was not kept. For each molecular bin of ``CarbonLayout`` and each
test cell of ``check_pah_grain_seam`` (Mathis background at the cell's G0, no RT photons), a four-state charge chain (z
= -1, 0, +1, +2) is built twice, from the PAH group table and from the 'ac' grain table at the bin's mean N_C, with
per-charge ingredients:
- absorption A_z [s^-1 per particle at G0 = 1] (PAH: A_bg; ac: pi a^2 Q_abs over the Mathis field),
- yield Y_z = K_z / A_z (photoemission or photodetachment per absorbed photon),
- energy per photoelectron E_z = H_z / K_z [eV],
- electron capture alpha_z(T) (z = 0, +1, +2),
- ion channels (PAH: associative detachment of anions by H, mutual neutralisation, charge transfer; ac: ion capture by
  negative and neutral grains with the Draine & Sutin focusing, sticking 1).
f(z) follows from the chain's detailed balance; heating per particle Gamma = G0 sum_z f(z) A_z Y_z E_z. Each chain is
checked against the full solves (the PAH table's ``PAHGroupTable.solve`` and ac's ``GroupChargingTable.solve``). Then,
starting from ac, one ingredient at a time is replaced by the PAH one (and the reverse), giving the factor each
ingredient contributes to the heating ratio.

Needs $CALIMA_PAHDB_DIR (PAHdb v4.00) and $CALIMA_DUSTEM_DIR.

    python diagnostics/gsd/check_seam_decomposition.py --max-mol 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import check_pah_grain_seam as seam

ING = ("absorption", "yield", "energy", "capture", "ions")
KB = 1.380649e-16
AMU = 1.66053906660e-24


def ac_ingredients(gt, a, T, n_e, n_h, ion, n_ion):
    from pycalima.models.dust_charge.shared_physics import DS87_J_function_vec

    A = seam.grain_absorption("ac", a)
    ing = dict(A=np.full(4, A), K=np.zeros(4), H=np.zeros(4), alpha=np.zeros(3), ion_up=np.zeros(3))
    for k, z in enumerate((-1, 0, 1, 2)):
        ing["K"][k], ing["H"][k] = seam.grain_bg(gt, a, z)
    for k, z in enumerate((0, 1, 2)):
        ing["alpha"][k] = seam.grain_capture(gt, a, z, T)
    m_i = seam.ION_MASS[ion] * AMU
    v_i = np.sqrt(8.0 * KB * T / (np.pi * m_i))
    for k, z in enumerate((-1, 0, 1)):            # positive ion capture moves z up by one
        J = float(DS87_J_function_vec(float(z), 1.0, a, T))
        ing["ion_up"][k] = n_ion * np.pi * a * a * v_i * J
    return ing


def pah_ingredients(pt, T, n_e, n_h, ion, n_ion):
    t = pt.t
    ing = dict(A=np.asarray(t["A_bg"], float).copy(), K=np.asarray(t["K_bg"], float).copy(),
               H=np.asarray(t["H_bg"], float).copy(), alpha=pt._in_T("alpha", T).copy(), ion_up=np.zeros(3))
    j = list(t["ions"]).index(ion)
    k_mn = pt._in_T("k_mn", T)
    ing["ion_up"][0] = t["k_ad"] * n_h + n_ion * k_mn[j]
    ing["ion_up"][1] = n_ion * t["k_ct"][j]
    return ing


def chain_heating(ing, G0, n_e):
    """Heating per particle [eV/s] and f(z) of the four-state chain."""
    A, K, H = ing["A"], ing["K"], ing["H"]
    Y = np.where(A > 0, K / np.maximum(A, 1e-300), 0.0)
    E = np.where(K > 0, H / np.maximum(K, 1e-300), 0.0)
    up = G0 * A[:3] * Y[:3] + ing["ion_up"]
    down = n_e * ing["alpha"]
    lnr = np.log(np.maximum(up, 1e-300)) - np.log(np.maximum(down, 1e-300))
    lnf = np.concatenate([[0.0], np.cumsum(lnr)])
    f = np.exp(lnf - lnf.max())
    f /= f.sum()
    return float(G0 * np.sum(f * A * Y * E)), f


def swap(base, other, name):
    """``base`` with ingredient ``name`` taken from ``other`` (yield and energy keep the base's absorption)."""
    out = {k: np.array(v, float).copy() for k, v in base.items()}
    if name == "absorption":
        Yb = base["K"] / np.maximum(base["A"], 1e-300)
        Eb = base["H"] / np.maximum(base["K"], 1e-300)
        out["A"] = np.array(other["A"], float)
        out["K"] = out["A"] * Yb
        out["H"] = out["K"] * Eb
    elif name == "yield":
        Yo = other["K"] / np.maximum(other["A"], 1e-300)
        Eb = base["H"] / np.maximum(base["K"], 1e-300)
        out["K"] = base["A"] * Yo
        out["H"] = out["K"] * Eb
    elif name == "energy":
        Eo = other["H"] / np.maximum(other["K"], 1e-300)
        out["H"] = base["K"] * Eo
    elif name == "capture":
        out["alpha"] = np.array(other["alpha"], float)
    elif name == "ions":
        out["ion_up"] = np.array(other["ion_up"], float)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-mol", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/carbon_plan/carbon_transfer_2026-11-23/seam_decomposition.txt")
    args = ap.parse_args(argv)
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.pahchem import population as pop
    from pycalima.pahchem.group_tables import PAHGroupTable

    edges = seam.bins(CarbonLayout())
    lib = pop.calibrate_thresholds(pop.library_v400())
    pahs = seam.pah_tables(lib, edges, args.max_mol, args.workers)
    lines = ["Seam decomposition (today's pahchem defaults, calibrated thresholds): heating per particle of the four-state",
             "chains against the full solves, and the factor on the 'ac' heating when one ingredient is taken from the PAH",
             "table (ac <- PAH), and on the PAH heating when taken from ac (PAH <- ac)."]
    for (lo, hi), pah in zip(edges, pahs):
        if pah is None:
            continue
        nc = pah["tab"]["nc"]
        a, tab, gt = seam.grain("ac", nc)
        pt = PAHGroupTable(pah["tab"])
        full = seam.solve_cells(pah, gt, a)
        lines.append(f"\n=== bin N_C {lo:.0f}-{hi:.0f} (mean N_C {nc:.0f}, a = {a * 1e7:.2f} nm)")
        lines.append("cell    G_PAH/G_ac full (chain)   chain/full PAH, ac   f_PAH(-1..+2) | f_ac   "
                     + "  ".join(f"{n:>10s}" for n in ING) + "   [ac <- PAH]; [PAH <- ac] below")
        for name, (T, n_h, n_e, ion, G0) in seam.CELLS.items():
            ip = pah_ingredients(pt, T, n_e, n_h, ion, n_e)
            ia = ac_ingredients(gt, a, T, n_e, n_h, ion, n_e)
            gp, fp = chain_heating(ip, G0, n_e)
            ga, fa = chain_heating(ia, G0, n_e)
            ev = 1.602176634e-12
            rp = gp * ev / full[name]["pah_gamma"]
            ra = ga * ev / full[name]["gr_gamma"]
            f_ac_p = [chain_heating(swap(ia, ip, n), G0, n_e)[0] / ga for n in ING]
            f_p_ac = [chain_heating(swap(ip, ia, n), G0, n_e)[0] / gp for n in ING]
            lines.append(f"{name:6s}  {full[name]['pah_gamma'] / full[name]['gr_gamma']:6.2f} ({gp / ga:5.2f})"
                         f"        {rp:5.2f}, {ra:5.2f}     "
                         + " ".join(f"{x:.2f}" for x in fp) + " | " + " ".join(f"{x:.2f}" for x in fa) + "   "
                         + "  ".join(f"{x:10.2f}" for x in f_ac_p))
            lines.append(" " * 78 + "  ".join(f"{x:10.2f}" for x in f_p_ac))
        # the ingredients themselves at G0 = 1, 300 K (z = -1..+2; capture z = 0..+2)
        ip = pah_ingredients(pt, 300.0, 1e-2, 1.0, "C+", 1e-2)
        ia = ac_ingredients(gt, a, 300.0, 1e-2, 1.0, "C+", 1e-2)
        for lab, ing in (("PAH", ip), ("ac ", ia)):
            Y = ing["K"] / np.maximum(ing["A"], 1e-300)
            E = ing["H"] / np.maximum(ing["K"], 1e-300)
            lines.append(f"   {lab} A/C " + " ".join(f"{x / nc:.2e}" for x in ing["A"]) + "  Y " + " ".join(
                f"{x:.3f}" for x in Y) + "  E[eV] " + " ".join(f"{x:.2f}" for x in E) + "  alpha(300K) "
                + " ".join(f"{x:.2e}" for x in ing["alpha"]))
    txt = "\n".join(lines)
    print(txt)
    Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
