#!/usr/bin/env python3
"""One-zone driver of the per-cell GSD operator (``pycalima.gsd.cell``, A3c).

``oracle``: the five A0 cases of ``check_bin_convergence.py`` (EA, 8 bins per composition), run step by step
with A0's own split step and with ``cell.update`` given the same models (A0's materials, pair tables, accretion
sticking and constant-rate sputtering; charging, sinks and the overflow bin off). The two must agree to
round-off: the operator adds nothing of its own. Reported: the largest difference of any bin's N and M relative to
the composition's total (empty bins make relative differences per bin meaningless), the relative difference of the
gas at the end, and the operator's element conservation.

    python diagnostics/gsd/run_cell.py oracle [--steps 25] [--out results/gsd_a3c/run_cell_oracle.txt]

``demo``: one-zone runs with every process on, in four cells (WNM, CNM, a dense core, a PDR; the PDR at G0 = 1e3):
- carbon on ``CarbonLayout`` with the 1-10 um overflow bin, the pair rule, area-node charging with the seam's sticking
  law, Coulomb explosion (fragments), sputtering and accretion of C (with the grains' Coulomb factor for the ionised
  part); the 4 molecular bins take no accretion, classical sputtering or grain explosion (their loss comes with the
  molecular tables, phase 6), so here they change only by the pair rule and by grains eroded into them;
- silicate on 8 bins with its overflow bin, the same processes but the pair rule;
- start: THEMIS J13-like distributions (carbon: plaw-ed + lognormal; silicate: lognormal), half of C and Si in dust.
Each step is precool then update. The dust temperature is given (20 K; precool computes it in a later step).
Reported per cell: the carbon shares (molecular bins, nano-grains, grains, gas) at the start and the end, the
photoelectric heating and recombination cooling at the end, element conservation, the overflow share, the charge
solves per cell (= bins) and the Python time per precool and per update.

With ``--molecular DIR`` the molecular bins take their tables (``gsd.molecular.MolecularTables``): the carbon loss
of pahchem's hydrogen tables in DIR (``--hot DIR``: the hot-gas set above 1e4 K) and, with ``--groups DIR``, their
charge, heating and cooling from the group tables (``build_molecular_group_tables.py``) and the photo-release S_eff of
molecules on any larger particle (``--y-pd``: the photodesorption yield, 'calibrated', default 0). With
``--optics`` precool computes the dust temperature per bin and the covalent a-C(:H) bins' equilibrium band gap, with
which their charge is solved (``gsd.optics.material_optics``; tables cached on first use).

    python diagnostics/gsd/run_cell.py demo [--steps 40] [--out results/gsd_a3c/run_cell_demo.txt]
        [--molecular DIR [--hot DIR] [--groups DIR] [--y-pd Y]] [--optics]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import check_bin_convergence as A0
import numpy as np

YR, PC = A0.YR, A0.PC          # A0's own constants, so that both runs see the same L and dt


def oracle_case(case, n_bins=8, scheme="ea", steps=25):
    from pycalima.gsd import cell

    cs = A0.CASES[case]
    ph, processes = A0.PHASES[cs["phase"]], cs["processes"]
    nH = ph["n_H"]
    a0, ours, gas = [], [], {}
    for mat in A0.materials(3.5e-8, 4.5e-8):
        g = A0.make_grid(mat, n_bins)
        N, M = A0.initial_bins(g, mat, cs["ic"])
        a0.append(A0.Composition(mat, g, scheme, M.copy(), N.copy()))
        gas[mat.key] = A0.SOLAR[mat.key] * (1.0 - A0.F_DEP0)
        erosion = (lambda T, n, m=mat, r=g.rho: A0.erosion(m, cs["phase"], r)) if "sputtering" in processes else None
        ours.append(cell.Composition(g, N * nH, M * nH, mat.key, mat.per_gram, mat.shatter, method=A0.method_of(scheme),
                                     sputtering=erosion, sublimation=mat.sublimation if "accretion" in processes
                                     else None, a_frag_min=mat.a_min))
    opt = cell.Options(processes=frozenset(processes) & cell.PROCESSES, errmax=A0.ERRMAX, n_speed=A0.N_SPEED,
                       table_ppd=A0.TABLE_PPD, threads=1)
    gas_ours = {k: v * nH for k, v in gas.items()}
    dt = cs["myr"] * 1e6 * YR / steps
    stats = {"coll_substeps": 0, "t_coll": 0.0, "t_acc": 0.0, "t_sput": 0.0}
    cons = 0.0
    for _ in range(steps):
        A0.step(a0, gas, cs["phase"], dt, processes, 1, stats)
        inp = cell.CellInputs(T=ph["T"], n_H=nH, n_e=0.0, gas=gas_ours, T_d=A0.T_DUST, v_turb=ph["V"] * 1e5,
                              L=ph["L"] * PC)
        res = cell.update(ours, inp, None, dt, opt)
        gas_ours = res.gas
        cons = max(cons, max(abs(v) for v in res.conservation.values()))
    dev = 0.0
    for c0, c1 in zip(a0, ours, strict=True):
        for x0, x1 in ((c0.N * nH, c1.N), (c0.M * nH, c1.M)):
            dev = max(dev, float(np.max(np.abs(x1 - x0)) / np.sum(np.abs(x0))))
    dgas = max(abs(gas_ours[k] / (gas[k] * nH) - 1.0) for k in gas)
    return dev, dgas, cons


M_H = 1.6735575e-24
AMU = 1.66053906660e-24
#: name: (T [K], n_H [cm^-3], n_e [cm^-3], G0 [Habing], v_turb [km/s], L [pc], duration [Myr], neutral fraction of the
#: gas-phase C and Si (labelled: ionised where the UV reaches, neutral in the dense core))
CELLS = {"WNM": (6000.0, 0.3, 0.012, 1.0, 20.0, 100.0, 30.0, 0.0), "CNM": (100.0, 30.0, 0.03, 1.0, 2.6, 10.0, 30.0, 0.0),
         "dense": (10.0, 1e4, 1e-3, 1e-2, 1.5, 1.0, 1.0, 1.0), "PDR": (300.0, 1e4, 1.0, 1e3, 1.5, 1.0, 1.0, 0.0)}
SOLAR = {"C": 2.69e-4, "Si": 3.24e-5}
#: n(H I) / n_H of the demo cells (labelled: atomic WNM/CNM, molecular core, half-atomic PDR layer)
N_HI_FRAC = {"WNM": 1.0, "CNM": 1.0, "dense": 1e-3, "PDR": 0.5}


def j13_like(grid, parts, mass_per_h, n_H):
    """N [cm^-3], M [g cm^-3] per bin of a distribution of the given shapes holding mass_per_h (dust mass per H mass)."""
    a = np.geomspace(grid.a_edges[0], grid.a_edges[-1], 4000)
    lna, m = np.log(a), grid.mass(a)
    dn = np.zeros(a.size)
    for w, f in parts:
        y = f(a) * a
        dn += w * y / np.trapezoid(y * m, lna)
    N, M = np.zeros(grid.n_bins), np.zeros(grid.n_bins)
    for i in range(grid.n_bins):
        sel = (a >= grid.a_edges[i]) & (a <= grid.a_edges[i + 1])
        N[i], M[i] = np.trapezoid(dn[sel], lna[sel]), np.trapezoid((dn * m)[sel], lna[sel])
    scale = mass_per_h * M_H * n_H / M.sum()
    return N * scale, M * scale


def covalent_bins(grid, n_mol, a_max=53.2e-7):
    """The covalent a-C(:H) bins: above the molecular ones, up to 53 nm."""
    return [i for i in range(n_mol, grid.n_bins) if grid.a_edges[i + 1] <= a_max]


def demo_compositions(n_H, molecular=None, hot=None, groups=None, optics=False):
    from dataclasses import replace

    from pycalima.gsd import cell
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.charging import AreaNodeCharging
    from pycalima.gsd.grid import MassGrid
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.pairs import PairRule
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    lay = CarbonLayout(overflow_a_max_cm=1e-3)
    cg = lay.grid()
    plaw = lambda a: np.where(a > 10e-7, a ** -5.0 * np.exp(-(a - 10e-7) / 50e-7), a ** -5.0)    # noqa: E731
    logn = lambda a0, s: lambda a: np.exp(-np.log(a / a0) ** 2 / (2 * s * s)) / a                   # noqa: E731
    Nc, Mc = j13_like(cg, [(0.73, plaw), (0.27, logn(7e-7, 1.0))], 2.33e-3, n_H)
    Nc[-1] = Mc[-1] = 0.0
    sil = get_material("silicate")
    m = 4.0 / 3.0 * np.pi * sil.rho * np.array([4.5e-8, 1e-4]) ** 3
    sg = MassGrid(np.geomspace(m[0], m[1], 9), sil.rho).with_overflow(1e-3)
    Ns, Ms = j13_like(sg, [(1.0, logn(8e-7, 1.0))], 5.1e-3, n_H)
    Ns[-1] = Ms[-1] = 0.0
    sub_c = replace(get_sublimation_model("graphite", "GD89"), alpha_gd89=0.3, density_gcm3=cg.rho)
    sub_s = replace(get_sublimation_model("silicate", "GD89"), alpha_gd89=0.3, density_gcm3=sg.rho)
    per_c = 1.0 / (12.011 * AMU)
    per_si = sub_s.key_per_unit / (sub_s.unit_mass_amu * AMU)
    gap_bins = covalent_bins(cg, lay.n_pah_bins) if optics else ()
    carbon = cell.Composition(cg, Nc, Mc, "C", per_c, get_material("themis_ac").shatter, sputtering="graphite",
                              sublimation=sub_c, pair_rule=PairRule.from_layout(lay, cg), molecular_bins=lay.n_pah_bins,
                              charging=AreaNodeCharging("themis_ac", cg, radii_per_bin=5, sticking="seam",
                                                        gap_bins=gap_bins))
    silicate = cell.Composition(sg, Ns, Ms, "Si", per_si, get_material("silicate").shatter, sputtering="silicate",
                                sublimation=sub_s, charging=AreaNodeCharging("silicate", sg, radii_per_bin=5))
    if molecular is not None:
        from pycalima.gsd.molecular import MolecularTables

        carbon.molecular_tables = MolecularTables.from_directories(molecular, hot, group_dir=groups)
    if optics:
        from pycalima.gsd.charging import AGN_GROUPS
        from pycalima.gsd.optics import covalent_hydrogen, material_optics

        carbon.optics = material_optics("themis_ac", cg, AGN_GROUPS, gap_bins)
        carbon.hydrogen = covalent_hydrogen(cg, gap_bins)
        silicate.optics = material_optics("silicate", sg, AGN_GROUPS)
    return carbon, silicate, lay


def carbon_shares(c, lay, gas_C):
    """Shares of the carbon in the molecular bins, the nano-grains (to 20 nm), the grains and the gas."""
    nb_mol = lay.n_pah_bins
    nano = (np.arange(c.grid.n_bins) >= nb_mol) & (c.grid.a_edges[1:] <= 20e-7 * 1.01)
    tot = (c.M.sum() + c.overflow_M) * c.per_gram + gas_C
    return np.array([c.M[:nb_mol].sum() * c.per_gram, c.M[nano].sum() * c.per_gram,
                     (c.M[nb_mol:].sum() - c.M[nano].sum() + c.overflow_M) * c.per_gram, gas_C]) / tot


def demo_cell(name, steps, molecular=None, hot=None, groups=None, y_pd=0.0, optics=False):
    import time

    from pycalima.gsd import cell

    T, n_H, n_e, G0, V, L, myr, x_n = CELLS[name]
    carbon, silicate, lay = demo_compositions(n_H, molecular, hot, groups, optics)
    gas = {"C": 0.5 * SOLAR["C"] * n_H, "Si": 0.5 * SOLAR["Si"] * n_H}
    opt = cell.Options(table_ppd=None, y_pd=y_pd)
    dt = myr * 1e6 * 3.15576e7 / steps
    start = carbon_shares(carbon, lay, gas["C"])
    t_pre = t_up = 0.0
    cons = 0.0
    for _ in range(steps):
        inp = cell.CellInputs(T=T, n_H=n_H, n_e=n_e, gas=gas, T_d=20.0, v_turb=V * 1e5, L=L * PC, G0_bg=G0,
                              neutral={"C": x_n, "Si": x_n}, n_HI=N_HI_FRAC[name] * n_H)
        t0 = time.perf_counter()
        pre = cell.precool([carbon, silicate], inp, opt)
        t1 = time.perf_counter()
        res = cell.update([carbon, silicate], inp, pre, dt, opt)
        t2 = time.perf_counter()
        t_pre, t_up = t_pre + t1 - t0, t_up + t2 - t1
        gas = res.gas
        cons = max(cons, max(abs(v) for v in res.conservation.values()))
    end = carbon_shares(carbon, lay, gas["C"])
    extra = None
    if optics:
        mb = lay.n_pah_bins
        extra = {"td_c": pre.t_dust[0][mb:carbon.grid.n_bins - 1], "td_s": pre.t_dust[1][:-1], "gap": pre.e_gap[0]}
    return dict(extra=extra, start=start, end=end, heating=pre.total_heating, cooling=pre.total_cooling, cons=cons,
                overflow=max(res.overflow_share), solves=carbon.grid.n_bins + silicate.grid.n_bins,
                t_pre=1e3 * t_pre / steps, t_up=1e3 * t_up / steps, myr=myr, n_H=n_H)


def cmd_demo(args):
    lines = ["One-zone runs of the GSD operator, every process on (precool + update per step); carbon shares",
             "(molecular N_C < 400, nano-grains to 20 nm, grains, gas) at the start and the end.",
             "cell   t[Myr]  start: mol   nano  grains  gas    end: mol   nano  grains  gas    Gamma/n_H   Lambda/n_H"
             "  [erg s^-1]  conservation  overflow  solves  ms/precool  ms/update"]
    if args.molecular:
        lines.insert(2, f"molecular tables: {args.molecular} (hot: {args.hot}), group tables: {args.groups}, "
                        f"Y_pd = {args.y_pd:g}")
    if args.optics:
        lines.insert(2, "optics on: T_d per bin and the covalent bins' equilibrium band gap from precool "
                        f"(n(H I)/n_H: {N_HI_FRAC})")
    extras = []
    for name in CELLS:
        r = demo_cell(name, args.steps, args.molecular, args.hot, args.groups, args.y_pd, args.optics)
        if r["extra"] is not None:
            x = r["extra"]
            extras.append(f"{name:6s} T_d carbon grain bins " + " ".join(f"{t:5.1f}" for t in x["td_c"])
                          + "  silicate " + " ".join(f"{t:5.1f}" for t in x["td_s"])
                          + "  E_g [eV] covalent bins " + " ".join(f"{e:4.2f}" for e in x["gap"]))
        lines.append(f"{name:6s} {r['myr']:5.1f}   " + " ".join(f"{x:6.3f}" for x in r["start"]) + "    "
                     + " ".join(f"{x:6.3f}" for x in r["end"])
                     + f"   {r['heating'] / r['n_H']:.2e}   {r['cooling'] / r['n_H']:.2e}            {r['cons']:.1e}"
                     f"      {r['overflow']:.1e}    {r['solves']}     {r['t_pre']:7.1f}    {r['t_up']:7.1f}")
        print(lines[-1], flush=True)
    if extras:
        lines += ["at the end, per bin (smallest first, overflow bins left out):", *extras]
    txt = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")
    print(txt)


def cmd_oracle(args):
    lines = ["A0 oracle: cell.update against check_bin_convergence.step (EA, 8 bins, same models), "
             f"{args.steps} steps per case.", "case            max |dN|/N_tot, |dM|/M_tot   gas rel    conservation"]
    for case in A0.CASES:
        dev, dgas, cons = oracle_case(case, steps=args.steps)
        lines.append(f"{case:18s} {dev:16.2e} {dgas:10.2e} {cons:12.2e}")
        print(lines[-1], flush=True)
    txt = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--steps", type=int, default=25)
    o.add_argument("--out", default="results/gsd_a3c/run_cell_oracle.txt")
    d = sub.add_parser("demo")
    d.add_argument("--steps", type=int, default=40)
    d.add_argument("--out", default="results/gsd_a3c/run_cell_demo.txt")
    d.add_argument("--molecular", default=None, help="hydrogen-table directory of the molecular bins (carbon loss)")
    d.add_argument("--hot", default=None, help="hot-gas hydrogen tables (above 1e4 K)")
    d.add_argument("--groups", default=None, help="group-table directory (charge, heating, photo-release)")
    d.add_argument("--y-pd", type=float, default=0.0)
    d.add_argument("--optics", action="store_true", help="precool computes T_d and the covalent band gaps")
    args = ap.parse_args(argv)
    {"oracle": cmd_oracle, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    main()
