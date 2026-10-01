"""Tabulated WDB06 grain charging with an EUV dimension (RAMSES charging_model='WDB06tab').

The original CALIMA tables give the charge of each bin as a function of
gamma = G0 sqrt(T) / n_e and T, for a Mathis ISRF (no ionising photons). This
extends them with the ionising photons: with n(H+) = n_e, the charge balance of
GroupChargingTable.solve depends on the cell only through

    T,   psi_F = G_FUV sqrt(T) / n_e,   psi_E = G_EUV sqrt(T) / n_e

where G_FUV is the 6-13.6 eV radiation in Habing units (RAMSES: advected_G0 +
rtz_UV_background_G0) with the Mathis ISRF shape (the table's K_bg kernels),
and G_EUV = c sum_{g: L0 >= 13.6 eV} N_g / F_EUV0 the ionising photon flux, with
the photon fractions of a reference blackbody (T_EUV) among the ionising groups.
Electron capture, ion capture, secondaries and pass-through all scale with
n_e sqrt(T) at fixed T, so per bin and grid point the table holds

    <Z>, sigma_Z, gamma_F, gamma_E, lambda, ln alpha_i

with Gamma = G_FUV gamma_F + G_EUV gamma_E (heating per unit G of each part at
the tabulated charge distribution: exact for any G, also outside the psi grid),
Lambda + Lambda_auto = n_e sqrt(T) lambda, and alpha_i the grain-assisted
recombination coefficients of RECOMB_IONS [cm^3 s^-1 per grain]. A lookup is
trilinear in (log T, log psi_F, psi_E), with psi_E linear between 0 and the
first logarithmic plane; inputs are clamped to the grid.

What it gives up against the per-group solve: the shape of the FUV/optical
spectrum (Mathis) and of the EUV (reference blackbody), He+ and He++ capture,
n(H+) != n_e, and interpolation.

    python -m pycalima.models.dust_charge.export_dust_charging_psitables
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg

FORMAT_VERSION = 1
LT = np.round(np.arange(1.0, 8.0 + 1e-9, 0.25), 6)            # log10 T [K]
LPSI_F = np.round(np.arange(-3.0, 9.0 + 1e-9, 0.25), 6)       # log10 psi_F [Habing K^1/2 cm^3]
LPSI_E = np.round(np.arange(-3.0, 9.0 + 1e-9, 0.25), 6)       # log10 psi_E; plus psi_E = 0 (first plane)
F_EUV0 = 1e8              # photons cm^-2 s^-1: unit of G_EUV (about the Habing photon flux)
T_EUV = 4e4               # K: reference blackbody of the ionising groups
EUV_EDGE = 13.6           # eV: groups with L0 >= this are ionising
QUANTITIES = ("Zmean", "Zsigma", "gamma_F", "gamma_E", "lambda") + tuple(f"lnalpha_{el}" for el, _, _ in rg.RECOMB_IONS)


def euv_shape(L0, L1, T_bb=T_EUV):
    """Photon fractions of a blackbody among the ionising groups (0 elsewhere) and
    the blackbody mean energy of every group [eV]."""
    w, egy = np.zeros(len(L0)), np.zeros(len(L0))
    kT = rg.KB_CGS * T_bb / rg.EV2ERG
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        E = np.geomspace(l0, l1, 400)
        n = E ** 2 / np.expm1(E / kT)
        w[g] = np.trapezoid(n, E) if l0 >= EUV_EDGE else 0.0
        egy[g] = np.trapezoid(n * E, E) / np.trapezoid(n, E)
    return w / w.sum(), egy


def _point(gt, w, T, psi_F, psi_E, Z_guess):
    """One grid point (n_e = 1): the charge solve and the tabulated quantities."""
    ne = 1.0
    G_F, G_E = psi_F * ne / np.sqrt(T), psi_E * ne / np.sqrt(T)
    N = G_E * F_EUV0 * w / rg.C_CGS
    ions = [(ne, 1.0, rg.M_HION)]
    r = gt.solve(N, rg.C_CGS, T, ne, ions, G0_bg=G_F, Z_guess=Z_guess)
    # heating per unit G of each part, on the same charge nodes
    gam = []
    for Nu, Gu in ((np.zeros_like(w), 1.0), (F_EUV0 * w / rg.C_CGS, 0.0)):
        gt._set_cell(Nu, rg.C_CGS, T, ne, ions, Gu)
        gam.append(sum(p * gt._heating_cooling(z)[0] for z, p in zip(r["nodes"], r["weights"])))
    gt._set_cell(N, rg.C_CGS, T, ne, ions, G_F)
    lam = (r["Lambda"] + r["Lambda_auto"]) / (ne * np.sqrt(T))
    vals = [r["Zmean"], r["Zsigma"], gam[0], gam[1], lam] + list(np.log(np.maximum(r["alpha_rec"], 1e-300)))
    return np.array(vals), r["Zstar"]


def _row(args):
    """All (psi_F, psi_E) points of one temperature (cold solves, the root search
    started from the previous psi_F)."""
    tab, L0, L1, T_bb, lt = args
    w, egy = euv_shape(L0, L1, T_bb)
    gt = rg.GroupChargingTable(tab, egy=egy, kernels="pchip")
    T = 10.0 ** lt
    psi_E = np.concatenate([[0.0], 10.0 ** LPSI_E])
    out = np.zeros((len(QUANTITIES), LPSI_F.size, psi_E.size))
    for k, pe in enumerate(psi_E):
        zg = None
        for j, lpf in enumerate(LPSI_F):
            out[:, j, k], zg = _point(gt, w, T, 10.0 ** lpf, pe, zg)
    return out


def build_psi_table(tab, T_bb=T_EUV, workers=1):
    """The table of one dust bin (a read_bin_table dict) on (LT, LPSI_F, [0, LPSI_E])."""
    L0, L1 = list(tab["L0"]), list(tab["L1"])
    args = [(tab, L0, L1, T_bb, lt) for lt in LT]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_row, args))
    else:
        rows = [_row(a) for a in args]
    w, _ = euv_shape(L0, L1, T_bb)
    return dict(version=FORMAT_VERSION, material=tab["material"], a=tab["a"], W=tab["W"], L0=np.array(L0),
                L1=np.array(L1), w=w, T_bb=T_bb, F0=F_EUV0, lT=LT, lpsiF=LPSI_F,
                psiE=np.concatenate([[0.0], 10.0 ** LPSI_E]), q=np.stack(rows, axis=1))   # (nq, nT, nPF, nPE)


# ---------------------------------------------------------------------------
# File format (read by RAMSES, calima/dust_charging_rtgroups.f90)
#   3 comment lines starting with '#'
#   version material_code a_cm W_eV nGroups nT nPF nPE T_bb F0
#   L0(1:nG) / L1(1:nG) / w(1:nG): EUV photon fractions (0 for the other groups)
#   log10 T(1:nT) / log10 psi_F(1:nPF) / psi_E(1:nPE): 0, then 10**LPSI_E (linear values)
#   for each of the QUANTITIES: nT * nPF lines of nPE values (T outer, psi_F, psi_E inner)
# ---------------------------------------------------------------------------

def write_psi_table(pt, path, label=""):
    path = Path(path)
    nq, nT, nPF, nPE = pt["q"].shape
    with open(path, "w") as fh:
        fh.write(f"# pyCALIMA WDB06 grain charging table with an EUV dimension, format {pt['version']}\n")
        fh.write(f"# bin {label}: {pt['material']}, a = {pt['a']:.6e} cm; quantities {' '.join(QUANTITIES)}\n")
        fh.write("# see pycalima.models.dust_charge.charging_psi_tables for the layout\n")
        fh.write(f"{pt['version']} {rg.MATERIAL_CODE[pt['material']]} {pt['a']:.12e} {pt['W']:.12e} "
                 f"{pt['L0'].size} {nT} {nPF} {nPE} {pt['T_bb']:.6e} {pt['F0']:.6e}\n")
        for key in ("L0", "L1", "w", "lT", "lpsiF", "psiE"):
            fh.write(rg._line(pt[key]))
        for iq in range(nq):
            for it in range(nT):
                for j in range(nPF):
                    fh.write(rg._line(pt["q"][iq, it, j]))
    return path


def read_psi_table(path):
    with open(path) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    it = iter(rows)
    head = next(it).split()
    version, mcode = int(head[0]), int(head[1])
    a, W = float(head[2]), float(head[3])
    nG, nT, nPF, nPE = (int(x) for x in head[4:8])
    T_bb, F0 = float(head[8]), float(head[9])
    vec = lambda: np.array(next(it).split(), dtype=float)          # noqa: E731
    pt = dict(version=version, material={v: k for k, v in rg.MATERIAL_CODE.items()}[mcode], a=a, W=W, T_bb=T_bb, F0=F0,
              L0=vec(), L1=vec(), w=vec(), lT=vec(), lpsiF=vec(), psiE=vec())
    q = np.zeros((len(QUANTITIES), nT, nPF, nPE))
    for iq in range(len(QUANTITIES)):
        for i in range(nT):
            for j in range(nPF):
                q[iq, i, j] = vec()
    pt["q"] = q
    return pt


def cell_G(Np, c_red, egy, L0, G0_bg, F0=F_EUV0):
    """G_FUV (RAMSES advected_G0: groups with 5.6 < egy < 13.6 eV, Habing; plus the
    background) and G_EUV (ionising photon flux / F0) of a cell."""
    Np, egy, L0 = np.asarray(Np, float), np.asarray(egy, float), np.asarray(L0, float)
    sel = (egy > 5.6) & (egy < 13.6)
    G_F = float(np.sum(Np[sel] * c_red * egy[sel]) * rg.EV2ERG / rg.HABING_FLUX) + G0_bg
    G_E = float(np.sum(Np[L0 >= EUV_EDGE]) * c_red / F0)
    return G_F, G_E


def _index(x, grid):
    f = (min(max(x, grid[0]), grid[-1]) - grid[0]) / (grid[1] - grid[0])
    i = min(int(f), grid.size - 2)
    return i, f - i


def psi_lookup(pt, T, ne, G_F, G_E):
    """<Z>, sigma_Z, Gamma, Lambda (incl. autoionisation) [erg/s per grain] and alpha_i
    of one bin from its table (trilinear, clamped)."""
    sq = np.sqrt(T)
    i, u = _index(np.log10(T), pt["lT"])
    j, v = _index(np.log10(max(G_F * sq / max(ne, 1e-300), 1e-300)), pt["lpsiF"])
    pE = G_E * sq / max(ne, 1e-300)
    lpE = np.log10(pt["psiE"][1:])
    if pE <= pt["psiE"][1]:
        k, s = 0, pE / pt["psiE"][1]
    else:
        k, s = _index(np.log10(pE), lpE)
        k += 1
    q = pt["q"]
    c = np.zeros(q.shape[0])
    for di, wi in ((0, 1 - u), (1, u)):
        for dj, wj in ((0, 1 - v), (1, v)):
            for dk, wk in ((0, 1 - s), (1, s)):
                c += wi * wj * wk * q[:, i + di, j + dj, k + dk]
    return dict(Zmean=c[0], Zsigma=c[1], Gamma=G_F * c[2] + G_E * c[3], Lambda=ne * sq * c[4], alpha_rec=np.exp(c[5:]))
