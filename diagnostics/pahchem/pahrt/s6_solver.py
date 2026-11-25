"""PAHrt S6: the per-cell solver (``gsd.pahrt``) on pahchem's kernels (``pahchem.pahrt_kernels``) against the truth.

- **V1, kernels.** At each test environment, the kernels' rates are put together from the cell's group fluxes,
  slopes, gas and ions, and compared with pahchem's own network at that environment (``Truth.network``, the full
  field), process by process. The error is flow-weighted with the truth's steady state F:
  sum |K - T| F / sum T F. The outputs are then compared, full and nested.
- **V2, one molecule.** The solver against S1's saved truth on its 214 nodes, per molecule.

The cell's group inputs come from pahchem's field at each node: photon flux and mean photon energy per group (from
1 eV, pahchem's lowest).

    python diagnostics/pahchem/pahrt/s6_solver.py kernels --workers 3   # the study's molecules -> s6/kernels
    python diagnostics/pahchem/pahrt/s6_solver.py v1
    python diagnostics/pahchem/pahrt/s6_solver.py v2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path("results/pahrt_study/s6")
KDIR = OUT / "kernels"
#: V1's environments: every S1 phase under every S1 field shape, at the phase's G0 and at 10^2 x it; the power laws
#: (1-13.6 eV, slopes on the bases' nodes) are exactly representable, so their errors are the kernels' alone
V1_FIELDS = ("draine", "draine+old", "kurucz10000", "kurucz20000", "kurucz40000", "powerlaw1.0", "powerlaw3.0")
#: outputs scored, and those compared in ratio (the rest in difference)
SCORED = ("f_m1", "f_0", "f_p1", "f_p2", "z_mean", "heating", "cooling", "absorbed", "carbon", "h2", "h_uptake",
          "e_source", "alpha_Cp", "dnh", "intact", "bare")
IN_RATIO = ("heating", "cooling", "absorbed", "carbon", "h2", "h_uptake", "alpha_Cp")


def groups():
    from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1

    return np.asarray(AGN_GROUP_L0, float), np.asarray(AGN_GROUP_L1, float)


def group_inputs(env, L0, L1):
    """(F_g [cm^-2 s^-1], mean photon energy [eV]) of pahchem's field at ``env`` in each group, from 1 eV."""
    from pycalima.pahchem import population as P

    f = P._unit_field(P.field_shape(env))
    E, n = f.energy_ev, f.photon_flux * env.g0
    F, e = np.zeros(L0.size), np.sqrt(L0 * L1)
    for g, (lo, hi) in enumerate(zip(L0, L1, strict=True)):
        k = (E >= max(lo, P.E_PH_MIN_EV)) & (E <= hi)
        if k.sum() < 2:
            continue
        F[g] = np.trapezoid(n[k], E[k])
        if F[g] > 0:
            e[g] = np.trapezoid(n[k] * E[k], E[k]) / F[g]
    return F, e


def cell_of(phase, L0, L1):
    from pycalima.gsd.pahrt import PAHrtCell

    env = phase.environment()
    F, e = group_inputs(env, L0, L1)
    return PAHrtCell(F, e, phase.T, phase.n_e, phase.n_hi, tuple(phase.ions)), env


def solver_outputs(o, z_top=2):
    """The solver's outputs under ``common.OUTPUTS``' names."""
    out = {"f_m1": o.get("f(-1)", 0.0), "f_0": o.get("f(+0)", 0.0), "f_p1": o.get("f(+1)", 0.0),
           "f_p2": o.get("f(+2)", 0.0), "alpha_Cp": o.get("alpha(C+)", 0.0), "alpha_Hp": o.get("alpha(H+)", 0.0)}
    for k in ("z_mean", "heating", "cooling", "absorbed", "carbon", "h2", "h_uptake", "e_source", "bare",
              "dehydrogenated", "intact", "superh", "dnh"):
        out[k] = o[k]
    return out


#: the bins scored (``--bins``; the largest molecules' bases take ~10 min each)
BINS = (0, 1, 2, 3)


def test_molecules():
    """The middle molecule of each bin's five (bins ``BINS``)."""
    return [ms[len(ms) // 2] for b, ms in C.molecules(5).items() if b in BINS]


# ---------------------------------------------------------------------------------------------------------- kernels
def _build(mol_name):
    from pycalima.pahchem import pahrt_kernels as PK

    mols = {m.name: m for ms in C.molecules(5).values() for m in ms}
    mol = mols[mol_name]
    path = KDIR / f"{PK.slug(mol)}.npz"
    if path.exists():
        return mol_name, 0.0
    t = time.time()
    L0, L1 = groups()
    PK.build_molecule(mol, L0, L1).save(path)
    return mol_name, time.time() - t


def kernels(workers, which):
    from concurrent.futures import ProcessPoolExecutor

    KDIR.mkdir(parents=True, exist_ok=True)
    mols = test_molecules() if which == "test" else [m for b, ms in C.molecules(5).items() if b in BINS for m in ms]
    names = [m.name for m in sorted(mols, key=lambda m: -m.nc)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for name, s in ex.map(_build, names):
                print(f"{name}: {s:.0f} s", flush=True)
    else:
        for name in names:
            print(f"{_build(name)[0]}: done", flush=True)


def load_solver(mol):
    from pycalima.gsd.pahrt import MoleculeSolver
    from pycalima.pahchem import pahrt_kernels as PK

    return MoleculeSolver(PK.MoleculeKernels.load(KDIR / f"{PK.slug(mol)}.npz"))


# ---------------------------------------------------------------------------------------------------------------- V1
def v1_environments():
    from dataclasses import replace

    out = []
    for name, ph in C.PHASES.items():
        for f in V1_FIELDS:
            for x in (1.0, 100.0):
                if ph.g0 * x <= 1e6:                     # pahchem's k_H tables end at G0 1e6
                    out.append((f"{name} {f} x{x:g}", replace(ph, field=f, g0=ph.g0 * x)))
    return out


def truth_rates(net, ms):
    """{proc key: rate[Z, N]} of pahchem's network on the solver's states."""
    iz = [list(ms.z).index(z) for z in net.charges]
    out = {}
    for p in net.processes:
        r = out.setdefault((p.name, int(p.dz), int(p.dnh), int(p.h2_released)), np.zeros((ms.z.size, ms.nh.size)))
        r[iz, :len(net.nh_values)] += p.rate
    return out


def v1():
    L0, L1 = groups()
    envs = v1_environments()
    rep = []
    res = {}
    for mol in test_molecules():
        ms = load_solver(mol)
        truth = C.Truth(mol)
        errs = {}
        outs = []
        for label, ph in envs:
            cell, env = cell_of(ph, L0, L1)
            net, W = truth.network(env)
            Ft = np.zeros((ms.z.size, ms.nh.size))
            iz = [list(ms.z).index(z) for z in net.charges]
            Ft[iz, :len(net.nh_values)] = net.steady_state_closed()
            to = C.outputs(net, W, mol.nh, truth.ion_rates(env), env)
            Kp, _ = ms.photon(cell)
            Kc, Ki, _ = ms.collisions(cell)
            K = Kp + Kc + sum((n * k for n, k in Ki.values()), np.zeros_like(Kp))
            T = truth_rates(net, ms)
            for p, key in enumerate(ms.mk.procs):
                tr = T.get(key, np.zeros_like(K[p]))
                den = float((tr * Ft).sum())
                num = float((np.abs(K[p] - tr) * Ft).sum())
                if den > 0 or num > 0:
                    errs.setdefault(key[0], []).append((label, num / den if den > 0 else np.inf, den))
            _, of = ms.solve(cell, "full")
            _, on = ms.solve(cell, "nested")
            outs.append((label, to, solver_outputs(of), solver_outputs(on)))
        res[mol.name] = (errs, outs, mol.nc)
        rep.append(f"\n=== {mol.name} (N_C {mol.nc}) ===")
        rep.append("process: flow-weighted relative error of the kernels' rates, median / max over the "
                   f"{len(envs)} environments (only where the process flows)")
        for name, v in sorted(errs.items(), key=lambda kv: -max(e for _, e, d in kv[1] if d > 0) if any(
                d > 0 for _, _, d in kv[1]) else 0):
            e = np.array([x for _, x, d in v if d > 0])
            if e.size:
                worst = max((x, lab) for lab, x, d in v if d > 0)
                rep.append(f"  {name:48s} {np.median(e):9.2e} {e.max():9.2e}  worst: {worst[1]}")
        rep.append("output: solver / truth (ratio outputs) or solver - truth; full | nested; median |dev| and worst")
        for k in SCORED:
            for m, col in (("full", 2), ("nested", 3)):
                d = np.array([dev(o[1][k], o[col][k], k, mol.nc) for o in outs])
                i = int(np.nanargmax(np.abs(d)))
                rep.append(f"  {k:10s} {m:6s} median {np.nanmedian(np.abs(d)):9.3g}  worst {d[i]:+9.3g} "
                           f"({outs[i][0]})")
    text = "\n".join(rep)
    (OUT / "v1_kernels.txt").write_text(text)
    print(text)
    np.save(OUT / "v1_results.npy", dict(res), allow_pickle=True)
    fig_v1(res, envs)


#: per-molecule rates below these are negligible (one event per molecule in 10 Gyr; carbon: N_C atoms in 10 Gyr):
#: both sides are floored there before the ratio, so that ratios of negligible rates do not dominate the scores
TEN_GYR_S = 3.156e17


def floor_of(k, nc):
    if k == "carbon":
        return nc / TEN_GYR_S
    if k in ("h2", "h_uptake"):
        return 1.0 / TEN_GYR_S
    return 0.0


def dev(t, s, k, nc=0):
    """log10(s / t) for ratio outputs (floored, ``floor_of``; 0 where both vanish), s - t otherwise."""
    if k in IN_RATIO:
        f = max(floor_of(k, nc), 1e-300)
        if t <= f and s <= f:
            return 0.0
        return float(np.log10(max(s, f) / max(t, f)))
    return float(s - t)


def fig_v1(res, envs):
    import matplotlib.pyplot as plt
    import plots

    plots.setup()
    mols = list(res)
    fig, axes = plt.subplots(len(mols), 2, figsize=(13, 3.2 * len(mols)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    labels = [lab for lab, _ in envs]
    for r, name in enumerate(mols):
        errs, outs, nc = res[name]
        procs = [p for p, v in errs.items() if any(d > 0 for _, _, d in v)]
        M = np.full((len(procs), len(labels)), np.nan)
        for i, p in enumerate(procs):
            for lab, e, d in errs[p]:
                if d > 0:
                    M[i, labels.index(lab)] = np.log10(max(e, 1e-8))
        ax = axes[r, 0]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=-6, vmax=0)
        ax.set_yticks(range(len(procs)), [p[:34] for p in procs], fontsize=6)
        ax.set_xticks(range(0, len(labels), 2), [labels[i].replace(" x1", "") for i in range(0, len(labels), 2)],
                      rotation=90, fontsize=5)
        ax.set_title(f"{name}: kernels vs pahchem, log10 flow-weighted rate error", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax = axes[r, 1]
        D = np.array([[dev(o[1][k], o[3][k], k, nc) for o in outs] for k in SCORED])
        im = ax.imshow(np.clip(D, -1, 1), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(len(SCORED)), SCORED, fontsize=6)
        ax.set_xticks([])
        ax.set_xlabel("the same environments", fontsize=7)
        ax.set_title("nested solver vs truth: log10 ratio (rates) or difference", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "v1_kernels.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- V2
def v2():
    import s1_drivers as S1

    L0, L1 = groups()
    nodes = S1.design()
    rep, res = [], {}
    for mol in [m for ms in C.molecules(5).values() for m in ms]:
        try:
            ms = load_solver(mol)
        except FileNotFoundError:
            continue
        d = np.load(S1.OUT / f"{S1.safe(mol.name)}.npz")
        truth = d["res"]
        D = np.full((len(nodes), len(SCORED)), np.nan)
        for i, (_, _, _, ph) in enumerate(nodes):
            if not np.isfinite(truth[i]).all():
                continue
            cell, _ = cell_of(ph, L0, L1)
            try:
                _, o = ms.solve(cell, "nested")
            except Exception as exc:                       # one node, not the run
                print(f"{mol.name} node {i}: {exc}")
                continue
            so = solver_outputs(o)
            t = dict(zip(C.OUTPUTS, truth[i], strict=True))
            D[i] = [dev(t[k], so[k], k, mol.nc) for k in SCORED]
        res[mol.name] = D
        rep.append(f"{mol.name:16s} " + " ".join(f"{k}:{np.nanmedian(np.abs(D[:, j])):.2g}/{np.nanmax(np.abs(D[:, j])):.2g}"
                                                 for j, k in enumerate(SCORED)))
    text = "median/max |log10 ratio| (rates) or |difference| over S1's nodes, nested solver vs truth\n" + "\n".join(rep)
    (OUT / "v2_molecules.txt").write_text(text)
    print(text)
    np.save(OUT / "v2_results.npy", res, allow_pickle=True)
    fig_v2(res, nodes)


def fig_v2(res, nodes):
    """Median and worst |deviation| per molecule and output, and the worst per S1 input swept."""
    import matplotlib.pyplot as plt
    import plots

    plots.setup()
    names = list(res)
    inputs = list(dict.fromkeys(n[1] for n in nodes))
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.35 * len(names) + 2.5))
    for ax, stat, title in ((axes[0], np.nanmedian, "median"), (axes[1], np.nanmax, "worst")):
        M = np.array([[stat(np.abs(res[n][:, j])) for j in range(len(SCORED))] for n in names])
        im = ax.imshow(np.log10(np.maximum(M, 1e-4)), aspect="auto", cmap="magma_r", vmin=-4, vmax=1)
        ax.set_xticks(range(len(SCORED)), SCORED, rotation=70, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"{title} |dev| over S1's nodes (log10)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    M = np.array([[np.nanmax(np.abs(np.concatenate([res[n][i:i + 1, :] for n in names
                                                     for i in range(len(nodes)) if nodes[i][1] == inp])[:, j]))
                   for j in range(len(SCORED))] for inp in inputs])
    im = axes[2].imshow(np.log10(np.maximum(M, 1e-4)), aspect="auto", cmap="magma_r", vmin=-4, vmax=1)
    axes[2].set_xticks(range(len(SCORED)), SCORED, rotation=70, fontsize=7)
    axes[2].set_yticks(range(len(inputs)), inputs, fontsize=8)
    axes[2].set_title("worst |dev| per input swept (all molecules)", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    fig.suptitle("PAHrt V2: nested solver vs pahchem; log10 ratio for rates, difference for shares and charge",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "v2_molecules.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    global BINS
    os.environ.setdefault("CALIMA_WORKERS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("kernels", "v1", "v2"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--molecules", choices=("test", "all"), default="test")
    ap.add_argument("--bins", type=int, nargs="+", default=list(BINS))
    a = ap.parse_args(argv)
    BINS = tuple(a.bins)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "kernels":
        kernels(a.workers, a.molecules)
    elif a.what == "v1":
        v1()
    else:
        v2()


if __name__ == "__main__":
    main()
