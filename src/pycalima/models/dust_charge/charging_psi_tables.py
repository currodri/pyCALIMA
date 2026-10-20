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
multilinear in (log T, log psi_F, sqrt h, h_E1, h_E2) and cubic in log psi_E (linear between
psi_E = 0 and the first logarithmic plane, and next to the ends); inputs are clamped to the grid. The hardness axis is
uniform in sqrt(h): the charge of strongly positive grains rises steeply with the first
hard photons (h ~ 0.01-0.1), which a uniform h axis interpolated 7% low in <Z> and 25%
in sigma_Z. Every node is evaluated on the exact P(Z) (``exact_distribution``, every
integer charge), not on the solver's Gaussian of a wide P(Z), so the table is exact at
its nodes whatever the solver's closure.

The EUV is split at the He and He+ ionisation edges into three bands (13.6-24.59,
24.59-54.42, >= 54.42 eV), and two more axes give its shape (format 5):

    h_E1 = F(>= 24.59 eV) / F_EUV,    h_E2 = F(>= 54.42 eV) / F(>= 24.59 eV)

With a single reference EUV shape (the blackbody of T_EUV), the charge in the H II gas
of hard sources was off by up to 3x: H absorbs the 13.6-24.59 eV photons first, so the
EUV hardens with depth, and a non-thermal source has a different split altogether. The
two hardnesses bring the table's model (solved exactly at a cell's coordinates) to
within ~1% of the group solve on the slab cells of 25-100 kK blackbodies and a
nu^-1.5 power law; the shape inside each band (e.g. 13.6-15.2 against 15.2-24.59 eV)
then no longer matters. <Z>, Gamma and alpha are close to linear in both: 3 nodes in
h_E1 is not enough for soft sources, though: a positive grain lit by a <= 43 kK
source is charged and heated mostly by the few photons above 24.59 eV (h_E1 ~ 0.02-0.13),
so <Z> and Gamma rise steeply over the first percent of h_E1 and then saturate. The h_E1
nodes (H_E1) are therefore 0, 0.02, 0.06, 0.15, 0.4, 1, log-interpolated below 0.15: along
h_E1 alone, <Z> within 2.7% and Gamma within 2.2% on cells of all seven test sources
(uniform 0, 0.5, 1: Gamma 30% off). h_E2 is close to linear: nodes 0, 1. A single node on
an axis fixes it at the reference blackbody's split (as format 4). Memory: 12x a table
without them (``--no-euv-hardness``).

What it gives up against the per-group solve: the shape inside each band (reference
blackbody), the in-band shape of the UV background, He+ and He++ capture,
n(H+) != n_e, and interpolation.

    python -m pycalima.models.dust_charge.export_dust_charging_psitables
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg

FORMAT_VERSION = 5
LT = np.round(np.arange(1.0, 8.0 + 1e-9, 0.25), 6)            # log10 T [K]
LPSI_F = np.round(np.arange(-3.0, 9.0 + 1e-9, 0.25), 6)       # log10 psi_F [K^1/2 cm^3, G in units of F0]
SQRT_H = np.round(np.arange(0.0, 0.6 + 1e-9, 0.1), 6)         # sqrt of the FUV hardness (h <= 0.36; Rayleigh-Jeans 0.39)
LPSI_E = np.round(np.arange(-3.0, 9.0 + 1e-9, 0.5), 6)        # log10 psi_E; plus psi_E = 0 (first plane)
H_E1 = np.array([0.0, 0.02, 0.06, 0.15, 0.4, 1.0])            # F(>= 24.59 eV) / F_EUV (not equally spaced)
HE_LOG_MAX = 0.15     # h_E segments with both nodes in (0, HE_LOG_MAX] are interpolated in log h_E
H_E2 = np.array([0.0, 1.0])                                    # F(>= 54.42 eV) / F(>= 24.59 eV)
# Grid: interpolation errors measured against direct solves at random points of the domain and
# on 2D Stromgren cells; coarser psi_F (1/3, 0.5 dex) or psi_E (1 dex) roughly double the <Z> tails.
F_EUV0 = 1e8              # photons cm^-2 s^-1: unit of G_FUV and G_EUV (about the Habing photon flux)
T_EUV = 4e4               # K: reference blackbody for the photons within each band
SOFT, HARD, EUV, EUV_HE, EUV_HEP = 1, 2, 3, 4, 5  # group roles (0: not used); EUV: 13.6-24.59 eV,
#                                                  EUV_HE: 24.59-54.42 eV, EUV_HEP: >= 54.42 eV
HE_EDGE, HEP_EDGE = 24.59, 54.42                   # eV: He and He+ ionisation
QUANTITIES = ("Zmean", "Zsigma", "lngamma_F", "lngamma_E", "lambda") + tuple(f"lnalpha_{el}" for el, _, _ in rg.RECOMB_IONS)


def group_roles(L0, L1, tol=1e-6):
    """Role of every group from its edges: SOFT inside 5.6-11.2 eV, HARD inside 11.2-13.6 eV,
    EUV from 13.6 eV, EUV_HE from 24.59 eV, EUV_HEP from 54.42 eV, 0 otherwise."""
    roles = np.zeros(len(L0), dtype=int)
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        if l0 >= HEP_EDGE - tol:
            roles[g] = EUV_HEP
        elif l0 >= HE_EDGE - tol:
            roles[g] = EUV_HE
        elif l0 >= 13.6 - tol:
            roles[g] = EUV
        elif l0 >= 11.2 - tol and l1 <= 13.6 + tol:
            roles[g] = HARD
        elif l0 >= 5.6 - tol and l1 <= 11.2 + tol:
            roles[g] = SOFT
    return roles


def band_shapes(L0, L1, T_bb=T_EUV):
    """Roles, the blackbody photon fraction of each group within its band, and the
    blackbody mean energy of every group [eV] (the default group energies);
    reference_hardness gives the same blackbody's split of the EUV bands."""
    roles = group_roles(L0, L1)
    n, egy = np.zeros(len(L0)), np.zeros(len(L0))
    kT = rg.KB_CGS * T_bb / rg.EV2ERG
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        E = np.geomspace(l0, l1, 400)
        ph = E ** 2 / np.expm1(E / kT)
        n[g] = np.trapezoid(ph, E)
        egy[g] = np.trapezoid(ph * E, E) / n[g]
    w = np.zeros(len(L0))
    for role in (SOFT, HARD, EUV, EUV_HE, EUV_HEP):
        sel = roles == role
        if sel.any():
            w[sel] = n[sel] / n[sel].sum()
    return roles, w, egy


def reference_hardness(L0, L1, T_bb=T_EUV):
    """h_E1, h_E2 of the T_bb blackbody (the value of an axis with a single node)."""
    roles = group_roles(L0, L1)
    kT = rg.KB_CGS * T_bb / rg.EV2ERG
    n = np.zeros(len(L0))
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        E = np.geomspace(l0, l1, 400)
        n[g] = np.trapezoid(E ** 2 / np.expm1(np.minimum(E / kT, 700.0)), E)
    return _hardness(n, roles)


def _hardness(N, roles):
    """h_E1, h_E2 of group photon numbers N (0 where the bands are empty)."""
    n1, n2, n3 = (N[roles == r].sum() for r in (EUV, EUV_HE, EUV_HEP))
    return (n2 + n3) / (n1 + n2 + n3) if n1 + n2 + n3 > 0 else 0.0, n3 / (n2 + n3) if n2 + n3 > 0 else 0.0


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


def photon_densities(roles, w, G_F, h, G_E, hE1, hE2):
    """Group photon densities [cm^-3] of the table's spectrum (c_red = c). Without a group
    in a band, its share goes to the one below (h_E2 into EUV_HE, h_E1 into EUV)."""
    has = {r: bool(np.any(roles == r)) for r in (EUV_HE, EUV_HEP)}
    f3 = hE1 * hE2 if has[EUV_HEP] else 0.0
    f2 = (hE1 - f3) if has[EUV_HE] else 0.0
    frac = np.select([roles == SOFT, roles == HARD, roles == EUV, roles == EUV_HE, roles == EUV_HEP],
                     [(1.0 - h) * G_F, h * G_F, (1.0 - f2 - f3) * G_E, f2 * G_E, f3 * G_E], 0.0)
    return frac * F_EUV0 * w / rg.C_CGS


def _point(gt, roles, w, T, psi_F, h, psi_E, hE1, hE2, Z_guess):
    """One grid point (n_e = 1): the charge solve and the tabulated quantities."""
    ne = 1.0
    G_F, G_E = psi_F * ne / np.sqrt(T), psi_E * ne / np.sqrt(T)
    ions = [(ne, 1.0, rg.M_HION)]
    N = photon_densities(roles, w, G_F, h, G_E, hE1, hE2)
    r = gt.solve(N, rg.C_CGS, T, ne, ions, Z_guess=Z_guess, recomb=None)
    Z, P = (r["nodes"], r["weights"]) if r["discrete"] else gt.exact_distribution(r["Zstar"], r["Zsigma"])
    zmean = float(P @ Z)
    zsig = float(np.sqrt(max(P @ Z ** 2 - zmean ** 2, 0.0)))
    lam = P @ gt._heating_cooling_arr(Z)[1]
    auto = gt._averages(Z[:1], P[:1], True)[2]          # autoionisation at Zmin, if P(Z) reaches it
    alpha = gt._recombination(Z, P)
    gam = []        # heating per unit G of each part, on the same charges
    for Nu in (photon_densities(roles, w, 1.0, h, 0.0, hE1, hE2), photon_densities(roles, w, 0.0, h, 1.0, hE1, hE2)):
        gt._set_cell(Nu, rg.C_CGS, T, ne, ions, 0.0)
        gam.append(P @ gt._heating_cooling_arr(Z)[0])
    gt._set_cell(N, rg.C_CGS, T, ne, ions, 0.0)
    vals = [zmean, zsig, np.log(max(gam[0], 1e-300)), np.log(max(gam[1], 1e-300)), (lam + auto) / (ne * np.sqrt(T))] \
        + list(np.log(np.maximum(alpha, 1e-300)))
    return np.array(vals), r["Zstar"]


def _row(args):
    """All (psi_F, h, h_E1, h_E2, psi_E) points of one temperature (cold solves, the root
    search started from the previous psi_F)."""
    tab, T_bb, egy, lt, grid = args
    roles, w, _ = band_shapes(tab["L0"], tab["L1"], T_bb)
    gt = rg.GroupChargingTable(tab, egy=egy, kernels="pchip")
    T = 10.0 ** lt
    psi_E = np.concatenate([[0.0], 10.0 ** grid["lpsiE"]])
    out = np.zeros((len(QUANTITIES), grid["lpsiF"].size, grid["sh"].size, grid["hE1"].size, grid["hE2"].size,
                    psi_E.size))
    for k, pe in enumerate(psi_E):
        for m, sh in enumerate(grid["sh"]):
            for e1, hE1 in enumerate(grid["hE1"]):
                for e2, hE2 in enumerate(grid["hE2"]):
                    zg = None
                    for j, lpf in enumerate(grid["lpsiF"]):
                        out[:, j, m, e1, e2, k], zg = _point(gt, roles, w, T, 10.0 ** lpf, sh ** 2, pe, hE1, hE2, zg)
    return out


def build_psi_table(tab, T_bb=T_EUV, egy=None, workers=1, grid=None, euv_hardness=True):
    """The table of one dust bin (a read_bin_table dict) on (LT, LPSI_F, SQRT_H, H_E1, H_E2,
    [0, LPSI_E]), or on grid (dict of lT, lpsiF, sh, hE1, hE2, lpsiE arrays, equally spaced),
    at the group mean energies egy [eV] (default: those of the T_bb blackbody). With
    euv_hardness False, or a run without groups beyond an edge, an EUV hardness axis has one
    node, at the T_bb blackbody's split."""
    egy = band_shapes(tab["L0"], tab["L1"], T_bb)[2] if egy is None else np.asarray(egy, float)
    if grid is None:
        roles = group_roles(tab["L0"], tab["L1"])
        ref = reference_hardness(tab["L0"], tab["L1"], T_bb)
        on1 = euv_hardness and (np.any(roles == EUV_HE) or np.any(roles == EUV_HEP))
        on2 = euv_hardness and np.any(roles == EUV_HE) and np.any(roles == EUV_HEP)
        grid = dict(lT=LT, lpsiF=LPSI_F, sh=SQRT_H, hE1=H_E1 if on1 else np.array([ref[0]]),
                    hE2=H_E2 if on2 else np.array([ref[1]]), lpsiE=LPSI_E)
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
                lT=np.array(grid["lT"]), lpsiF=np.array(grid["lpsiF"]), sh=np.array(grid["sh"]),
                hE1=np.array(grid["hE1"], float), hE2=np.array(grid["hE2"], float),
                psiE=np.concatenate([[0.0], 10.0 ** np.asarray(grid["lpsiE"])]),
                q=q)   # (nq, nT, nPF, nH, nHE1, nHE2, nPE)


# ---------------------------------------------------------------------------
# Files (read by RAMSES, calima/dust_charging_rtgroups.f90): a text header
# dust_charging_psitab_<bin>.dat and the values in dust_charging_psitab_<bin>.bin
#   header: 3 comment lines starting with '#'
#   version material_code a_cm W_eV nGroups nT nPF nH nHE1 nHE2 nPE T_bb F0
#   roles(1:nG) (0 unused, 1 soft FUV, 2 hard FUV, 3 EUV 13.6-24.59, 4 EUV 24.59-54.42, 5 EUV >= 54.42)
#   / w(1:nG) / L0(1:nG) / L1(1:nG) / egy(1:nG) [eV]
#   bg_soft bg_hard: Mathis photon flux at G0 = 1 in the two FUV bands [cm^-2 s^-1]
#   log10 T(1:nT) / log10 psi_F(1:nPF) / sqrt h(1:nH) / h_E1(1:nHE1) / h_E2(1:nHE2) /
#   psi_E(1:nPE): 0, then 10**LPSI_E
#   .bin: little-endian float32, the QUANTITIES in order, each nT x nPF x nH x nHE1 x nHE2 x nPE
#         with psi_E fastest (Fortran q(nPE, nHE2, nHE1, nH, nPF, nT, nq))
# ---------------------------------------------------------------------------

def write_psi_table(pt, path, label=""):
    path = Path(path)
    nq, nT, nPF, nH, nE1, nE2, nPE = pt["q"].shape
    with open(path, "w") as fh:
        fh.write(f"# pyCALIMA WDB06 grain charging table with FUV hardness and EUV axes, format {pt['version']}\n")
        fh.write(f"# bin {label}: {pt['material']}, a = {pt['a']:.6e} cm; values in {path.with_suffix('.bin').name}: "
                 f"{' '.join(QUANTITIES)}\n")
        fh.write("# see pycalima.models.dust_charge.charging_psi_tables for the layout\n")
        fh.write(f"{pt['version']} {rg.MATERIAL_CODE[pt['material']]} {pt['a']:.12e} {pt['W']:.12e} "
                 f"{pt['L0'].size} {nT} {nPF} {nH} {nE1} {nE2} {nPE} {pt['T_bb']:.6e} {pt['F0']:.6e}\n")
        fh.write(" ".join(str(int(r)) for r in pt["roles"]) + "\n")
        for key in ("w", "L0", "L1", "egy", "bg", "lT", "lpsiF", "sh", "hE1", "hE2", "psiE"):
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
    nG, nT, nPF, nH, nE1, nE2, nPE = (int(x) for x in head[4:11])
    T_bb, F0 = float(head[11]), float(head[12])
    vec = lambda: np.array(next(it).split(), dtype=float)          # noqa: E731
    pt = dict(version=version, material={v: k for k, v in rg.MATERIAL_CODE.items()}[mcode], a=a, W=W, T_bb=T_bb, F0=F0,
              roles=vec().astype(int), w=vec(), L0=vec(), L1=vec(), egy=vec(), bg=vec(), lT=vec(), lpsiF=vec(), sh=vec(), hE1=vec(), hE2=vec(),
              psiE=vec())
    pt["q"] = np.fromfile(path.with_suffix(".bin"), dtype="<f4").astype(np.float64).reshape(
        len(QUANTITIES), nT, nPF, nH, nE1, nE2, nPE)
    return pt


def cell_inputs(pt, Np, c_red, G0_bg):
    """G_FUV, h, G_EUV, h_E1, h_E2 of a cell from its group photon densities and the UV background."""
    Np = np.asarray(Np, float)
    F_s = c_red * Np[pt["roles"] == SOFT].sum() + G0_bg * pt["bg"][0]
    F_h = c_red * Np[pt["roles"] == HARD].sum() + G0_bg * pt["bg"][1]
    F_F = F_s + F_h
    euv = pt["roles"] >= EUV
    hE1, hE2 = _hardness(Np, pt["roles"])
    return F_F / pt["F0"], (F_h / F_F if F_F > 0 else 0.0), c_red * Np[euv].sum() / pt["F0"], hE1, hE2


def _index(x, grid):
    if grid.size == 1:
        return 0, 0.0
    f = (min(max(x, grid[0]), grid[-1]) - grid[0]) / (grid[1] - grid[0])
    i = min(int(f), grid.size - 2)
    return i, f - i


CUBIC_PSI_E = [QUANTITIES.index(k) for k in ("Zmean", "Zsigma", "lambda")]
LOG_QUANTITIES = [i for i in range(len(QUANTITIES)) if i not in CUBIC_PSI_E]
LN_FLOOR_CUBIC = np.log(1e-250)    # a log quantity with a node below this (a shut channel) stays linear


def _hE_index(x, nodes):
    """Node and weight on an EUV hardness axis (not equally spaced; clamped): linear in h_E,
    or in log h_E where both nodes are in (0, HE_LOG_MAX]."""
    if nodes.size == 1:
        return 0, 0.0
    x = min(max(x, nodes[0]), nodes[-1])
    i = min(int(np.searchsorted(nodes, x, side="right")) - 1, nodes.size - 2)
    lo, hi = nodes[i], nodes[i + 1]
    if lo > 0.0 and hi <= HE_LOG_MAX:
        return i, np.log(x / lo) / np.log(hi / lo)
    return i, (x - lo) / (hi - lo)


def _psiE_weights(k, s, n):
    """Weights along psi_E from node k (0: the psi_E = 0 plane): cubic (4-point Lagrange) in
    log psi_E where the four planes k-1..k+2 are logarithmic, else linear. <Z> grows faster
    than linearly in log psi_E as the grain charges up, which linear interpolation on the
    0.5 dex planes overshot by up to ~20 charges (~15%) in hard-source H II gas, and ln alpha
    of the repelled ions with it. The ln gamma and ln alpha planes reach the 1e-300 floor where
    a channel shuts, which a cubic through them turns into overflows: there they stay linear
    (LN_FLOOR_CUBIC)."""
    if k >= 2 and k + 2 <= n - 1:
        return ((-1, -s * (s - 1) * (s - 2) / 6), (0, (s + 1) * (s - 1) * (s - 2) / 2),
                (1, -(s + 1) * s * (s - 2) / 2), (2, (s + 1) * s * (s - 1) / 6))
    return ((0, 1 - s), (1, s))


def psi_lookup(pt, T, ne, G_F, h, G_E, hE1, hE2):
    """<Z>, sigma_Z, Gamma, Lambda (incl. autoionisation) [erg/s per grain] and alpha_i
    of one bin from its table (multilinear, cubic along psi_E: _psiE_weights; clamped). Along
    the EUV hardness axes the heating is interpolated in gamma, not ln gamma: at fixed P(Z) it
    is linear in the split of the photons, and with a soft source (h_E1 ~ 0.02) a positive grain
    is heated almost only by the few photons above 24.59 eV, which ln gamma between the
    h_E1 = 0 and 0.5 nodes put 25-50% low."""
    sq = np.sqrt(T)
    i, u = _index(np.log10(T), pt["lT"])
    j, v = _index(np.log10(max(G_F * sq / max(ne, 1e-300), 1e-300)), pt["lpsiF"])
    m, x = _index(np.sqrt(max(h, 0.0)), pt["sh"])
    e1, y1 = _hE_index(hE1, pt["hE1"])
    e2, y2 = _hE_index(hE2, pt["hE2"])
    pE = G_E * sq / max(ne, 1e-300)
    if pE <= pt["psiE"][1]:
        k, s = 0, pE / pt["psiE"][1]
    else:
        k, s = _index(np.log10(pE), np.log10(pt["psiE"][1:]))
        k += 1
    q = pt["q"]
    corners = lambda i0, f, n: ((0, 1 - f), (1, f)) if n > 1 else ((0, 1.0),)   # noqa: E731
    use = np.zeros(q.shape[0], bool)
    use[CUBIC_PSI_E] = True
    c = np.zeros(q.shape[0])
    gam = np.zeros(2)
    for d1, w1 in corners(e1, y1, q.shape[4]):          # each EUV-hardness corner, then linear in h_E
        for d2, w2 in corners(e2, y2, q.shape[5]):
            lin = np.zeros(q.shape[0])
            cub = np.zeros(q.shape[0])
            low = np.full(q.shape[0], np.inf)           # lowest node value seen by the cubic
            for di, wi in corners(i, u, q.shape[1]):
                for dj, wj in corners(j, v, q.shape[2]):
                    for dm, wm in corners(m, x, q.shape[3]):
                        wa = wi * wj * wm
                        for dk, wk in corners(k, s, q.shape[6]):
                            lin += wa * wk * q[:, i + di, j + dj, m + dm, e1 + d1, e2 + d2, k + dk]
                        for dk, wk in _psiE_weights(k, s, q.shape[6]):
                            node = q[:, i + di, j + dj, m + dm, e1 + d1, e2 + d2, k + dk]
                            cub += wa * wk * node
                            low = np.minimum(low, node)
            use[LOG_QUANTITIES] = low[LOG_QUANTITIES] > LN_FLOOR_CUBIC
            lin[use] = cub[use]
            c += w1 * w2 * lin
            gam += w1 * w2 * np.exp(lin[2:4])           # heating: linear in gamma along h_E
    return dict(Zmean=c[0], Zsigma=c[1], Gamma=G_F * gam[0] + G_E * gam[1], Lambda=ne * sq * c[4],
                alpha_rec=np.exp(c[5:]))
