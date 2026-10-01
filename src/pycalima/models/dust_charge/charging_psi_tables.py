"""Tabulated WDB06 grain charging with FUV hardness and EUV axes (RAMSES charging_model='WDB06tab').

The original CALIMA tables give the charge of each bin as a function of
gamma = G0 sqrt(T) / n_e and T, for a Mathis ISRF (no ionising photons). This
extends them with the shape of the FUV and with the ionising photons: with
n(H+) = n_e, the charge balance of GroupChargingTable.solve depends on the cell
through

    T,   psi_F = G_FUV sqrt(T) / n_e,   h,   psi_E = G_EUV sqrt(T) / n_e

where G_FUV = F_FUV / F0 is the photon flux of the FUV groups (soft: inside
5.6-11.2 eV, hard: inside 11.2-13.6 eV, plus the UV background's photons in the
same bands), h = F_hard / F_FUV its hardness (the Mathis ISRF has h = 0.107, a
4e4 K blackbody 0.19, the Rayleigh-Jeans limit 0.39), and G_EUV = F_EUV / F0
that of the ionising groups (L0 >= 13.6 eV). Within each band the photons are
shared among its groups as in a reference blackbody (T_EUV); groups below 5.6 eV
are left out (they do not change the charge). The group kernels are taken at the
run's group mean energies (``egy``, RAMSES info_rt; they set the in-group
spectral shape: in the Stromgren test, blackbody energies instead of the run's
bias the heating by -13%), which RAMSES checks against the table. Photoemission is linear in the
group photon densities, so with these two FUV numbers the FUV part of the
rates is that of the cell's own FUV groups; in the Stromgren test this removes
the bias of a Mathis FUV (charge within 1%, heating within 1% at the median).

Electron and ion capture, secondaries and pass-through all scale with
n_e sqrt(T) at fixed T, so per bin and grid point the table holds

    <Z>, sigma_Z, ln gamma_F, ln gamma_E, lambda, ln alpha_i

with Gamma = G_FUV gamma_F + G_EUV gamma_E (heating per unit G of each part at
the tabulated charge distribution: exact for any G, also outside the psi grid;
interpolated in ln, as alpha_i: both fall by orders of magnitude on positive grains),
Lambda + Lambda_auto = n_e sqrt(T) lambda, and alpha_i the grain-assisted
recombination coefficients of RECOMB_IONS [cm^3 s^-1 per grain]. A lookup is
quadrilinear in (log T, log psi_F, h, psi_E), with psi_E linear between 0 and
the first logarithmic plane; inputs are clamped to the grid.

What it gives up against the per-group solve: the EUV shape (reference
blackbody), the in-band shape of the UV background, He+ and He++ capture,
n(H+) != n_e, and interpolation.

    python -m pycalima.models.dust_charge.export_dust_charging_psitables
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg

FORMAT_VERSION = 3
LT = np.round(np.arange(1.0, 8.0 + 1e-9, 0.25), 6)            # log10 T [K]
LPSI_F = np.round(np.arange(-3.0, 9.0 + 1e-9, 0.25), 6)       # log10 psi_F [K^1/2 cm^3, G in units of F0]
H_FUV = np.round(np.arange(0.0, 0.4 + 1e-9, 0.1), 6)          # FUV hardness (0.05: same errors, 1.8x the memory)
LPSI_E = np.round(np.arange(-3.0, 9.0 + 1e-9, 0.5), 6)        # log10 psi_E; plus psi_E = 0 (first plane)
# Grid: interpolation errors measured against direct solves at random points of the domain and
# on 2D Stromgren cells; coarser psi_F (1/3, 0.5 dex) or psi_E (1 dex) roughly double the <Z> tails.
F_EUV0 = 1e8              # photons cm^-2 s^-1: unit of G_FUV and G_EUV (about the Habing photon flux)
T_EUV = 4e4               # K: reference blackbody for the photons within each band
SOFT, HARD, EUV = 1, 2, 3  # group roles (0: not used)
QUANTITIES = ("Zmean", "Zsigma", "lngamma_F", "lngamma_E", "lambda") + tuple(f"lnalpha_{el}" for el, _, _ in rg.RECOMB_IONS)


def group_roles(L0, L1, tol=1e-6):
    """Role of every group from its edges: SOFT inside 5.6-11.2 eV, HARD inside 11.2-13.6 eV,
    EUV above 13.6 eV, 0 otherwise."""
    roles = np.zeros(len(L0), dtype=int)
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        if l0 >= 13.6 - tol:
            roles[g] = EUV
        elif l0 >= 11.2 - tol and l1 <= 13.6 + tol:
            roles[g] = HARD
        elif l0 >= 5.6 - tol and l1 <= 11.2 + tol:
            roles[g] = SOFT
    return roles


def band_shapes(L0, L1, T_bb=T_EUV):
    """Roles, the blackbody photon fraction of each group within its band, and the
    blackbody mean energy of every group [eV] (the default group energies)."""
    roles = group_roles(L0, L1)
    n, egy = np.zeros(len(L0)), np.zeros(len(L0))
    kT = rg.KB_CGS * T_bb / rg.EV2ERG
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        E = np.geomspace(l0, l1, 400)
        ph = E ** 2 / np.expm1(E / kT)
        n[g] = np.trapezoid(ph, E)
        egy[g] = np.trapezoid(ph * E, E) / n[g]
    w = np.zeros(len(L0))
    for role in (SOFT, HARD, EUV):
        sel = roles == role
        if sel.any():
            w[sel] = n[sel] / n[sel].sum()
    return roles, w, egy


def background_band_fluxes(L0, L1, roles):
    """Photon flux [cm^-2 s^-1] of the Mathis ISRF at G0 = 1 Habing (the K_bg normalisation)
    in the soft and hard FUV bands of the groups."""
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field
    Eb = np.geomspace(1.0, 13.6, 4000)
    uE = np.array([Mathis83_radiation_field(e) for e in Eb])
    band = Eb >= 6.0
    uE *= rg.HABING_FLUX / (rg.C_CGS * np.trapezoid(uE[band], Eb[band]))
    Fb = rg.C_CGS * uE / (Eb * rg.EV2ERG)
    out = []
    for role in (SOFT, HARD):
        sel = roles == role
        if not sel.any():
            out.append(0.0)
            continue
        lo, hi = min(np.asarray(L0)[sel]), max(np.asarray(L1)[sel])
        m = (Eb >= lo) & (Eb <= hi)
        out.append(float(np.trapezoid(Fb[m], Eb[m])))
    return np.array(out)


def photon_densities(roles, w, G_F, h, G_E):
    """Group photon densities [cm^-3] of the table's spectrum (c_red = c)."""
    frac = np.select([roles == SOFT, roles == HARD, roles == EUV], [(1.0 - h) * G_F, h * G_F, G_E], 0.0)
    return frac * F_EUV0 * w / rg.C_CGS


def _point(gt, roles, w, T, psi_F, h, psi_E, Z_guess):
    """One grid point (n_e = 1): the charge solve and the tabulated quantities."""
    ne = 1.0
    G_F, G_E = psi_F * ne / np.sqrt(T), psi_E * ne / np.sqrt(T)
    ions = [(ne, 1.0, rg.M_HION)]
    N = photon_densities(roles, w, G_F, h, G_E)
    r = gt.solve(N, rg.C_CGS, T, ne, ions, Z_guess=Z_guess)
    gam = []        # heating per unit G of each part, on the same charge nodes
    for Nu in (photon_densities(roles, w, 1.0, h, 0.0), photon_densities(roles, w, 0.0, h, 1.0)):
        gt._set_cell(Nu, rg.C_CGS, T, ne, ions, 0.0)
        gam.append(sum(p * gt._heating_cooling(z)[0] for z, p in zip(r["nodes"], r["weights"])))
    gt._set_cell(N, rg.C_CGS, T, ne, ions, 0.0)
    lam = (r["Lambda"] + r["Lambda_auto"]) / (ne * np.sqrt(T))
    vals = [r["Zmean"], r["Zsigma"], np.log(max(gam[0], 1e-300)), np.log(max(gam[1], 1e-300)), lam] \
        + list(np.log(np.maximum(r["alpha_rec"], 1e-300)))
    return np.array(vals), r["Zstar"]


def _row(args):
    """All (psi_F, h, psi_E) points of one temperature (cold solves, the root search
    started from the previous psi_F)."""
    tab, T_bb, egy, lt, grid = args
    roles, w, _ = band_shapes(tab["L0"], tab["L1"], T_bb)
    gt = rg.GroupChargingTable(tab, egy=egy, kernels="pchip")
    T = 10.0 ** lt
    psi_E = np.concatenate([[0.0], 10.0 ** grid["lpsiE"]])
    out = np.zeros((len(QUANTITIES), grid["lpsiF"].size, grid["h"].size, psi_E.size))
    for k, pe in enumerate(psi_E):
        for m, h in enumerate(grid["h"]):
            zg = None
            for j, lpf in enumerate(grid["lpsiF"]):
                out[:, j, m, k], zg = _point(gt, roles, w, T, 10.0 ** lpf, h, pe, zg)
    return out


def build_psi_table(tab, T_bb=T_EUV, egy=None, workers=1, grid=None):
    """The table of one dust bin (a read_bin_table dict) on (LT, LPSI_F, H_FUV, [0, LPSI_E]),
    or on grid (dict of lT, lpsiF, h, lpsiE arrays, equally spaced), at the group mean
    energies egy [eV] (default: those of the T_bb blackbody)."""
    egy = band_shapes(tab["L0"], tab["L1"], T_bb)[2] if egy is None else np.asarray(egy, float)
    grid = dict(lT=LT, lpsiF=LPSI_F, h=H_FUV, lpsiE=LPSI_E) if grid is None else grid
    args = [(tab, T_bb, egy, lt, grid) for lt in grid["lT"]]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_row, args))
    else:
        rows = [_row(a) for a in args]
    roles, w, _ = band_shapes(tab["L0"], tab["L1"], T_bb)
    q = np.stack(rows, axis=1).astype(np.float32).astype(np.float64)       # as stored
    return dict(version=FORMAT_VERSION, material=tab["material"], a=tab["a"], W=tab["W"],
                L0=np.asarray(tab["L0"], float), L1=np.asarray(tab["L1"], float), egy=egy, roles=roles, w=w,
                bg=background_band_fluxes(tab["L0"], tab["L1"], roles), T_bb=T_bb, F0=F_EUV0,
                lT=np.array(grid["lT"]), lpsiF=np.array(grid["lpsiF"]), h=np.array(grid["h"]),
                psiE=np.concatenate([[0.0], 10.0 ** np.asarray(grid["lpsiE"])]),
                q=q)   # (nq, nT, nPF, nH, nPE)


# ---------------------------------------------------------------------------
# Files (read by RAMSES, calima/dust_charging_rtgroups.f90): a text header
# dust_charging_psitab_<bin>.dat and the values in dust_charging_psitab_<bin>.bin
#   header: 3 comment lines starting with '#'
#   version material_code a_cm W_eV nGroups nT nPF nH nPE T_bb F0
#   roles(1:nG) (0 unused, 1 soft FUV, 2 hard FUV, 3 EUV) / w(1:nG) / L0(1:nG) / L1(1:nG) / egy(1:nG) [eV]
#   bg_soft bg_hard: Mathis photon flux at G0 = 1 in the two FUV bands [cm^-2 s^-1]
#   log10 T(1:nT) / log10 psi_F(1:nPF) / h(1:nH) / psi_E(1:nPE): 0, then 10**LPSI_E
#   .bin: little-endian float32, the QUANTITIES in order, each nT x nPF x nH x nPE with
#         psi_E fastest (Fortran q(nPE, nH, nPF, nT, nq))
# ---------------------------------------------------------------------------

def write_psi_table(pt, path, label=""):
    path = Path(path)
    nq, nT, nPF, nH, nPE = pt["q"].shape
    with open(path, "w") as fh:
        fh.write(f"# pyCALIMA WDB06 grain charging table with FUV hardness and EUV axes, format {pt['version']}\n")
        fh.write(f"# bin {label}: {pt['material']}, a = {pt['a']:.6e} cm; values in {path.with_suffix('.bin').name}: "
                 f"{' '.join(QUANTITIES)}\n")
        fh.write("# see pycalima.models.dust_charge.charging_psi_tables for the layout\n")
        fh.write(f"{pt['version']} {rg.MATERIAL_CODE[pt['material']]} {pt['a']:.12e} {pt['W']:.12e} "
                 f"{pt['L0'].size} {nT} {nPF} {nH} {nPE} {pt['T_bb']:.6e} {pt['F0']:.6e}\n")
        fh.write(" ".join(str(int(r)) for r in pt["roles"]) + "\n")
        for key in ("w", "L0", "L1", "egy", "bg", "lT", "lpsiF", "h", "psiE"):
            fh.write(rg._line(pt[key]))
    np.ascontiguousarray(pt["q"], dtype="<f4").tofile(path.with_suffix(".bin"))
    return path


def read_psi_table(path):
    path = Path(path)
    with open(path) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    it = iter(rows)
    head = next(it).split()
    version, mcode = int(head[0]), int(head[1])
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: format {version}, expected {FORMAT_VERSION} (re-export the tables)")
    a, W = float(head[2]), float(head[3])
    nG, nT, nPF, nH, nPE = (int(x) for x in head[4:9])
    T_bb, F0 = float(head[9]), float(head[10])
    vec = lambda: np.array(next(it).split(), dtype=float)          # noqa: E731
    pt = dict(version=version, material={v: k for k, v in rg.MATERIAL_CODE.items()}[mcode], a=a, W=W, T_bb=T_bb, F0=F0,
              roles=vec().astype(int), w=vec(), L0=vec(), L1=vec(), egy=vec(), bg=vec(), lT=vec(), lpsiF=vec(), h=vec(),
              psiE=vec())
    pt["q"] = np.fromfile(path.with_suffix(".bin"), dtype="<f4").astype(np.float64).reshape(
        len(QUANTITIES), nT, nPF, nH, nPE)
    return pt


def cell_inputs(pt, Np, c_red, G0_bg):
    """G_FUV, h, G_EUV of a cell from its group photon densities and the UV background."""
    Np = np.asarray(Np, float)
    F_s = c_red * Np[pt["roles"] == SOFT].sum() + G0_bg * pt["bg"][0]
    F_h = c_red * Np[pt["roles"] == HARD].sum() + G0_bg * pt["bg"][1]
    F_F = F_s + F_h
    return F_F / pt["F0"], (F_h / F_F if F_F > 0 else 0.0), c_red * Np[pt["roles"] == EUV].sum() / pt["F0"]


def _index(x, grid):
    f = (min(max(x, grid[0]), grid[-1]) - grid[0]) / (grid[1] - grid[0])
    i = min(int(f), grid.size - 2)
    return i, f - i


def psi_lookup(pt, T, ne, G_F, h, G_E):
    """<Z>, sigma_Z, Gamma, Lambda (incl. autoionisation) [erg/s per grain] and alpha_i
    of one bin from its table (quadrilinear, clamped)."""
    sq = np.sqrt(T)
    i, u = _index(np.log10(T), pt["lT"])
    j, v = _index(np.log10(max(G_F * sq / max(ne, 1e-300), 1e-300)), pt["lpsiF"])
    m, x = _index(h, pt["h"])
    pE = G_E * sq / max(ne, 1e-300)
    if pE <= pt["psiE"][1]:
        k, s = 0, pE / pt["psiE"][1]
    else:
        k, s = _index(np.log10(pE), np.log10(pt["psiE"][1:]))
        k += 1
    q = pt["q"]
    c = np.zeros(q.shape[0])
    for di, wi in ((0, 1 - u), (1, u)):
        for dj, wj in ((0, 1 - v), (1, v)):
            for dm, wm in ((0, 1 - x), (1, x)):
                for dk, wk in ((0, 1 - s), (1, s)):
                    c += wi * wj * wm * wk * q[:, i + di, j + dj, m + dm, k + dk]
    return dict(Zmean=c[0], Zsigma=c[1], Gamma=G_F * np.exp(c[2]) + G_E * np.exp(c[3]), Lambda=ne * sq * c[4],
                alpha_rec=np.exp(c[5:]))
