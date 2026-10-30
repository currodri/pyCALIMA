"""THEMIS a-C ('cm20') grain charging against DustEM v4.3's own charge distributions.

DustEM's ``out/ZDIST.RES`` holds the charge distributions of the 25 CM20 radii of
GRAIN_J13 (0.4-4900 nm) in its default gas (GAS.DAT: T = 8000 K, n_e = 7.2e-3,
n(H+) = 7.08e-3, n(C+) = 1.2e-4 cm^-3) and the Mathis ISRF (G0 = 1). The file shipped
with the release is wrong above ~0.5 um (<Z> ~ 0 for 1-5 um grains, where a fresh
build gives -119 to -2226) and differs from a fresh build by ~4% at the smallest
sizes; build one with ``build_dustem.sh`` and pass its file with ``--zdist``.

For each radius the script builds the RT-group table (``rt_group_charging``, material
'cm20'), solves the cell RAMSES would solve with no RT photons and the UV background
set to DustEM's field, takes the exact P(Z) over the integers, and compares <Z>,
sigma_Z, P(Z), the photoemission rate J_pe, the gas current J_gas, the absorbed power
P_abs and the photoelectric efficiency H_pe/P_abs. The WDB06 graphite charge of the
same radii is plotted for reference (``--no-graphite`` to skip it).

Known differences, by construction (see docs/physics/charging-cm20.md):
- WDB06's photo-excited secondaries, which DustEM lacks, add 1-3% to J_pe above
  ~10 nm and raise <Z> by up to half a sigma_Z; without them <Z> is within 0.1
  sigma_Z and J_pe within 1% (tests/test_cm20_charging.py);
- secondaries excited by gas electrons (DS79), which DustEM lacks, make 1-4 nm
  grains more positive in this warm gas: <Z> 8% higher at 1.3 nm;
- J_gas below ~2 nm: DustEM prints the electron current with the l_e of charge Z,
  while its P(Z) uses that of Z - 1 (as we do throughout);
- the table interpolates the kernels linearly in asinh U (401 nodes);
- above ~0.9 um DustEM's Q_CM20.DAT has Q_sca up to ~700 below 0.1 um, which enters
  Kimura's yield through Q_ext (the 2.2 um point).

``--material ac`` runs pyCALIMA's own a-C instead (WD01 thresholds without the
aromatic correction, WD01 sticking, recomputed large-grain optics), to show how
far it departs from DustEM.

Needs $CALIMA_DUSTEM_DIR (the extracted release). Prints a table; writes a figure to
the plots directory (``--out`` to change it).

    diagnostics/dust_charge/build_dustem.sh DUSTEM_RELEASE /tmp/dustem_build
    python diagnostics/dust_charge/check_cm20_dustem.py --zdist /tmp/dustem_build/out/ZDIST.RES
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.dust_charge import wdb06_yields as wy
from pycalima.models.dust_radiation import themis_optics

C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
AMU = 1.66053906660e-24
EV = 1.602176634e-12
GROUPS = ((5.6,), (13.6,))      # one group to the Lyman limit (unused: no RT photons)


def read_zdist(path, gtype="CM20"):
    """Blocks of ZDIST.RES for ``gtype``: header numbers and (Z, f) per radius, and the gas."""
    lines = Path(path).read_text().splitlines()
    out, k = [], 0
    keys = ("nzb", "a", "Zmin", "Zmax", "Zeq", "Zmean", "sdZ", "Jpe", "Jgas", "Pabs", "Cgas_Pabs", "Hpe_Pabs")
    while k < len(lines):
        f = lines[k].split()
        if len(f) == 14 and f[0] == "#" and f[1] == gtype:
            rec = dict(zip(keys, [int(f[2])] + [float(x) for x in f[3:]]))
            zf = np.array([lines[k + 1 + j].split() for j in range(rec["nzb"])], float)
            rec["Z"], rec["f"] = zf[:, 0], zf[:, 1]
            out.append(rec)
            k += 1 + rec["nzb"]
        else:
            k += 1
    i = next(j for j, l in enumerate(lines) if l.startswith("# Gas Summary"))
    T, nH, _, _, G0 = (float(x) for x in lines[i + 1].split())
    x, xp, xm = (float(v) for v in lines[i + 2].split()[:3])
    gas = dict(T=T, nH=nH, G0=G0, ne=x * nH, n_Hp=xp * nH, n_Cp=xm * nH)
    return out, gas


def dustem_isrf(root, G0):
    """DustEM ISRF.DAT: wavelengths [um], 4 pi I_nu [erg cm^-2 s^-1 Hz^-1], times G0."""
    rows = [l.split() for l in (Path(root) / "data" / "ISRF.DAT").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    d = np.array([r for r in rows[1:] if len(r) == 2], float)
    return d[:, 0], G0 * d[:, 1]


def habing_g0(lam_um, fnu):
    """Field in 6-13.6 eV over the Habing flux (1.6e-3 erg cm^-2 s^-1), interpolated
    linearly in wavelength as DustEM does (INTPOL), on a fine grid: the ISRF.DAT
    points alone miss the 12.7-13.6 eV interval (7% of the band)."""
    o = np.argsort(lam_um)
    E = np.geomspace(6.0, 13.6, 4000)
    lam = 1.23984193 / E
    nu = C_CGS / (lam * 1e-4)
    f = np.interp(lam, lam_um[o], fnu[o], left=0.0, right=0.0)
    return np.trapezoid(f, nu) / rg.HABING_FLUX


def p_abs(a_cm, lam_um, fnu):
    """Absorbed power pi a^2 int Q_abs 4 pi I_nu dnu [erg/s] (THEMIS Q_abs)."""
    q = themis_optics.efficiencies(a_cm * 1e4, lam_um)[0][:, 0]
    nu = C_CGS / (lam_um * 1e-4)
    o = np.argsort(nu)
    return np.pi * a_cm ** 2 * np.trapezoid((q * fnu)[o], nu[o])


def charge_rates(gt, Z, P):
    """J_pe, J_gas [s^-1] and H_pe [erg/s] over the charges Z with probabilities P,
    for the cell set by the last ``gt.solve``."""
    c = gt.cell
    jpe = jgas = hpe = 0.0
    for z, p in zip(Z, P):
        i, w = gt._u_index(z * rg.E2_EV_CM / gt.a)
        photo = c["G0_bg"] * gt._kern(gt.t["K_bg"], None, i, w)
        up, down, _ = gt._rates(z)
        jpe += p * photo
        jgas += p * (up - photo + down)
        hpe += p * gt._heating_cooling(z)[0]
    return jpe, jgas, hpe


def solve(material, a, gas, G0_bg):
    tab = rg.build_bin_table(material, a, *GROUPS)
    gt = rg.GroupChargingTable(tab)
    ions = [(gas["n_Hp"], 1.0, 1.00794 * AMU), (gas["n_Cp"], 1.0, 12.0107 * AMU)]
    r = gt.solve(np.zeros(1), C_CGS, gas["T"], gas["ne"], ions, G0_bg=G0_bg, recomb=None)
    Z, P = gt.exact_distribution(r["Zstar"], max(r["Zsigma"], 0.5))
    return gt, Z, P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zdist", default=None, help="ZDIST.RES (default: the release's out/ZDIST.RES)")
    ap.add_argument("--out", default=None, help="figure path")
    ap.add_argument("--no-graphite", action="store_true", help="skip the graphite reference")
    ap.add_argument("--material", default="cm20", choices=("cm20", "ac"),
                    help="'cm20' (the DustEM port, the check) or 'ac' (pyCALIMA's a-C: how far it departs)")
    args = ap.parse_args(argv)

    root = Path(os.environ.get("CALIMA_DUSTEM_DIR", ""))
    if not (root / "data" / "CHRG_CM20.DAT").exists():
        raise SystemExit("set $CALIMA_DUSTEM_DIR to the extracted DustEM release")
    recs, gas = read_zdist(args.zdist or root / "out" / "ZDIST.RES")
    lam, fnu = dustem_isrf(root, gas["G0"])
    G0_bg = habing_g0(lam, fnu)
    print(f"gas: T = {gas['T']:.0f} K, n_e = {gas['ne']:.3e}, n(H+) = {gas['n_Hp']:.3e}, "
          f"n(C+) = {gas['n_Cp']:.3e} cm^-3; DustEM G0 = {gas['G0']:g} -> {G0_bg:.4f} Habing")

    rows, dists = [], []
    hdr = ("a_nm", "Zmean_D", "Zmean", "sd_D", "sd", "Jpe/D", "Jgas/D", "Pabs/D", "eff_D", "eff/D", "TV")
    print(" ".join(f"{h:>9s}" for h in hdr))
    for rec in recs:
        a = rec["a"]
        gt, Z, P = solve(args.material, a, gas, G0_bg)
        zm = float(P @ Z)
        sd = float(np.sqrt(max(P @ Z ** 2 - zm ** 2, 0.0)))
        jpe, jgas, hpe = charge_rates(gt, Z, P)
        pa = p_abs(a, lam, fnu)
        zz = np.union1d(Z, rec["Z"])
        tv = 0.5 * np.abs(np.interp(zz, Z, P, 0, 0) - np.interp(zz, rec["Z"], rec["f"], 0, 0)).sum()
        row = (a * 1e7, rec["Zmean"], zm, rec["sdZ"], sd, jpe / rec["Jpe"], jgas / rec["Jgas"],
               pa / rec["Pabs"], rec["Hpe_Pabs"], hpe / pa / rec["Hpe_Pabs"], tv)
        rows.append(row)
        dists.append((a, Z, P, rec["Z"], rec["f"]))
        print(" ".join(f"{v:9.4g}" for v in row), flush=True)
    rows = np.array(rows)

    gra = None
    if not args.no_graphite:
        gra = []
        for rec in recs:
            _, Z, P = solve("graphite", rec["a"], gas, G0_bg)
            gra.append(float(P @ Z))
        gra = np.array(gra)

    print("\nmax |<Z> - <Z>_D| / max(sigma_D, 1): "
          f"{np.max(np.abs(rows[:, 2] - rows[:, 1]) / np.maximum(rows[:, 3], 1.0)):.3f}")
    for name, j in (("J_pe", 5), ("J_gas", 6), ("P_abs", 7), ("H_pe/P_abs", 9)):
        print(f"{name:>10s} ratio to DustEM: {rows[:, j].min():.3f} - {rows[:, j].max():.3f}")
    print(f"total-variation distance of P(Z): max {rows[:, 10].max():.3f}")
    plot(rows, dists, gra, gas, args.out, args.material)


def plot(rows, dists, gra, gas, out, material="cm20"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    a = rows[:, 0]
    ax = axs[0, 0]
    ax.plot(a, rows[:, 1], "k-", label="DustEM CM20")
    ax.plot(a, rows[:, 2], "o", ms=4, mfc="none", color="C0", label=f"pyCALIMA {material}")
    if gra is not None:
        ax.plot(a, gra, "--", color="C3", label="pyCALIMA graphite (WDB06)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"$a$ [nm]")
    ax.set_ylabel(r"$\langle Z \rangle$")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[0, 1]
    ax.plot(a, rows[:, 3], "k-", label="DustEM")
    ax.plot(a, rows[:, 4], "o", ms=4, mfc="none", color="C0", label="pyCALIMA")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a$ [nm]")
    ax.set_ylabel(r"$\sigma_Z$")
    ax = axs[1, 0]
    for j, lab in ((5, r"$J_{\rm pe}$"), (6, r"$J_{\rm gas}$"), (7, r"$P_{\rm abs}$"),
                   (9, r"$H_{\rm pe}/P_{\rm abs}$")):
        ax.plot(a, rows[:, j], "-o", ms=3, label=lab)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$a$ [nm]")
    ax.set_ylabel("pyCALIMA / DustEM")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1, 1]
    for k, idx in enumerate((0, 3, 6, 9)):
        if idx >= len(dists):
            continue
        a_k, Z, P, ZD, fD = dists[idx]
        ax.plot(ZD, fD, "-", color=f"C{k}", label=fr"$a$ = {a_k * 1e7:.2g} nm")
        ax.plot(Z, P, "o", ms=4, mfc="none", color=f"C{k}")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    ax.set_xlabel(r"$Z$")
    ax.set_ylabel(r"$P(Z)$ (lines: DustEM)")
    ax.legend(frameon=False, fontsize=9)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.suptitle(fr"CM20 charging, $T$ = {gas['T']:.0f} K, $n_e$ = {gas['ne']:.2g} cm$^{{-3}}$, "
                 fr"$G_0$ = {gas['G0']:g} (Mathis)", fontsize=11)
    fig.tight_layout()
    out = Path(out) if out else _paths.get_plots_dir() / "dust_charge" / "check_cm20_dustem.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
