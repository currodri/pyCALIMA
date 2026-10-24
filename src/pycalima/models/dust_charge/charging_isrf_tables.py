"""WDB06 grain charging under a uniform ISRF, tabulated for RAMSES charging_model='WDB06isrf':
the charge of runs without RT photon groups, with its PE heating, recombination cooling,
Coulomb focusing factors and grain-assisted ion recombination, and the PE heating of the
charging models that do not compute their own (WDB06rt does).

The charge balance is that of ``rt_group_charging`` with no local RT photons: the UV background
(the Mathis ISRF scaled to G0 Habing in 6-13.6 eV) and H+ ions with n(H+) = n_e. Every rate is
linear in G0 or in n_e, so one table at G0 = 1 serves any G0 through x = n_e/G0:

    <Z>, sigma_Z, D_j, alpha_i (G0, T, n_e) = table(T, x)
    Gamma (G0, T, n_e)             = G0 * Gamma_1(T, x)               [erg s^-1 per grain]
    Lambda (G0, T, n_e)            = n_e sqrt(T) * lambda_1(T, x)     [erg s^-1 per grain]

(Lambda is linear in n_e at G0 = 1, so Lambda/n_e keeps the right limit where x is clamped at
the table's edge, e.g. G0 -> 0. It is the net recombination cooling with autoionisation, which
is negative for strongly positive grains in cold gas, so lambda_1 is stored linearly.) Every
node is evaluated on the exact P(Z), every integer charge with P > 1e-300 of the maximum within
8 sigma + 10 of the peak and always from Z = -3 to +3, not on the solver's Gaussian of a wide
P(Z) nor on its trimmed window of a narrow one: <Z>, sigma_Z, the PE heating, the
recombination cooling, the Coulomb focusing factors D_j over that P(Z) for impactor
charges j = -1, +1, +2, +3 (``coulomb_focusing``), and the grain-assisted recombination
coefficients alpha_i [cm^3 s^-1 per grain] of the singly charged ions of
``rt_group_charging.RECOMB_IONS`` (``GroupChargingTable._recombination``: case-A charge
thresholds), read by RAMSES for dust_ion_recombination. Through the tail of P(Z) at the
opposite sign, D_j of a small grain in cold gas differs by orders of magnitude from any
Gaussian of the same mean and width.

Stored, on (log10 T, log10 x): <Z>, sigma_Z, ln Gamma_1, lambda_1, ln D_j, ln alpha_i. A lookup
is bilinear and clamped to the grid.

    python -m pycalima.models.dust_charge.export_dust_charging_isrf
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import coulomb_focusing as cf
from pycalima.models.dust_charge import rt_group_charging as rg

FORMAT_VERSION = 2
LT = np.linspace(1.0, 9.0, 161)         # log10 T [K]
LX = np.linspace(-8.0, 8.0, 161)        # log10 (n_e / G0) [cm^-3]
D_CHARGES = (-1, 1, 2, 3)               # impactor charges of the D planes
QUANTITIES = (("Zmean", "Zsigma", "lnGamma1", "lambda1") + tuple(f"lnD{j:+d}" for j in D_CHARGES)
              + tuple(f"lnalpha_{ion}" for ion, _, _ in rg.RECOMB_IONS))
LN_FLOOR = np.log(1e-300)


# Memory bounds of full_distribution (each kept charge costs ~1 kB downstream: heating, cooling, D_j,
# alpha_i). The shipped bins span at most ~4e4 charges and take the contiguous path, as before; a 1 mm
# grain has ~1e9 allowed charges, and the solver's sigma_Z of a wide P(Z) can be far wider than the
# exact one (a P(Z) pinned at Zmin).
MAX_CONTIGUOUS = 100_000     # charges evaluated at once from -3..+3 to the window
CHUNK = 100_000              # charges per step where ln P is accumulated piecewise
MAX_CHARGES = 50_000         # charges kept; a wider P(Z) is kept on every stride-th charge
MAX_MARCH = 50_000_000       # charges ln P may be accumulated over (time, not memory)


def _lnf(gt, lo, hi):
    """Integer charges lo..hi and ln P(Z) up to a constant (the cumulative g = ln up/down)."""
    Z = np.arange(lo, hi + 1.0)
    return Z, np.concatenate([[0.0], np.cumsum(gt._g_arr(Z[:-1]))])


def _march(gt, z, l, z_end):
    """Chunks (Zc, Lc) of ln P on the charges from z (exclusive; ln P(z) = l) to z_end (inclusive),
    from g at every charge, CHUNK at a time, in the direction of z_end."""
    d = 1.0 if z_end > z else -1.0
    done = 0
    while (z_end - z) * d > 0:
        n = int(min(CHUNK, abs(z_end - z)))
        done += n
        if done > MAX_MARCH:
            raise ValueError(f"P(Z) of a = {gt.t['a']:.3e} cm: ln P over more than {MAX_MARCH} charges")
        if d > 0:                                       # ln P(z + k) = ln P(z) + g(z) + ... + g(z + k - 1)
            Zc = np.arange(z + 1.0, z + n + 1.0)
            Lc = l + np.cumsum(gt._g_arr(Zc - 1.0))
        else:                                           # ln P(z - k) = ln P(z) - g(z - 1) - ... - g(z - k)
            Zc = np.arange(z - 1.0, z - n - 1.0, -1.0)
            Lc = l - np.cumsum(gt._g_arr(Zc))
        yield Zc, Lc
        z, l = Zc[-1], Lc[-1]


def _floor_end(gt, z, l, z_end, lfloor):
    """The last charge from z towards z_end with ln P > lfloor, and its ln P (ln P(z) = l > lfloor).
    Away from the peak P(Z) is log-concave (g decreases with Z), so ln P falls monotonically."""
    for Zc, Lc in _march(gt, z, l, z_end):
        k = np.nonzero(Lc <= lfloor)[0]
        if k.size:
            return (Zc[k[0] - 1], Lc[k[0] - 1]) if k[0] > 0 else (z, l)
        z, l = Zc[-1], Lc[-1]
    return z, l


def full_distribution(gt, Zs, width):
    """The exact P(Z) of the cell set by the last solve on every integer charge within
    8 width + 10 of Zs and from -3 to +3 (where D_j of the opposite sign and the photoemission
    of strongly charged grains live), kept down to 1e-300 of its maximum. Returns the charges,
    P, and the stride: 1, or, where P(Z) is above that floor over more than MAX_CHARGES charges
    (sigma_Z of 1e4 or more: large grains in hot gas), every stride-th charge, each standing for
    stride charges (ln P exact at each; sums over a P(Z) that wide do not see the stride)."""
    wlo = max(gt.t["Zmin"], np.floor(Zs - 8 * width - 10))
    whi = min(gt.t["Zmax"], np.ceil(Zs + 8 * width + 10))
    lo = max(gt.t["Zmin"], min(wlo, -3.0))
    hi = min(gt.t["Zmax"], max(whi, 3.0))
    if hi - lo + 1 <= MAX_CONTIGUOUS:
        Z, lnf = _lnf(gt, lo, hi)
        lnf -= lnf.max()
        k = lnf > LN_FLOOR
        P = np.exp(lnf[k])
        return Z[k], P / P.sum(), 1
    # Far from neutral, or wide, without an array over every charge in between: (1) the peak of
    # ln P in the window, (2) the charges above the floor around it, out towards -3..+3 while P
    # is above it, (3) ln P on those, on a stride if they are more than MAX_CHARGES
    zp, lp = wlo, 0.0
    for Zc, Lc in _march(gt, wlo, 0.0, whi):
        i = int(np.argmax(Lc))
        if Lc[i] > lp:
            zp, lp = Zc[i], Lc[i]
    lfloor = lp + LN_FLOOR
    za, la = _floor_end(gt, zp, lp, lo, lfloor)
    zb, _ = _floor_end(gt, zp, lp, hi, lfloor)
    stride = max(1, int(np.ceil((zb - za + 1) / MAX_CHARGES)))
    Zk, Lk = [np.array([za])], [np.array([la])]
    for Zc, Lc in _march(gt, za, la, zb):
        keep = np.mod(Zc - za, stride) == 0
        Zk.append(Zc[keep])
        Lk.append(Lc[keep])
    Z, lnf = np.concatenate(Zk), np.concatenate(Lk)
    lnf -= lnf.max()
    k = lnf > LN_FLOOR
    P = np.exp(lnf[k])
    return Z[k], P / P.sum(), stride


def _node(gt, T, x, Z_guess):
    """One node (G0 = 1, n_e = x): the solve, then everything on its exact P(Z)."""
    ions = [(x, 1.0, rg.M_HION)]
    r = gt.solve(np.zeros(gt.t["L0"].size), 0.0, T, x, ions, G0_bg=1.0, Z_guess=Z_guess, recomb=None)
    Z, P, stride = full_distribution(gt, r["Zstar"], max(r["Zsigma"], 0.5))
    zmean = float(P @ Z)
    zsig = float(np.sqrt(max(P @ Z ** 2 - zmean ** 2, 0.0)))
    heat, cool = gt._heating_cooling_arr(Z)
    auto = gt._averages(Z[:1], P[:1] / stride, True)[2]   # autoionisation at Zmin, if P(Z) reaches it
    D = cf.coulomb_factors_exact(Z, P, T, gt.t["a"], max(D_CHARGES))
    lnD = [np.log(max(D[j + 1], 1e-300)) for j in D_CHARGES]
    lnalpha = [np.log(max(v, 1e-300)) for v in gt._recombination(Z, P)]
    vals = [zmean, zsig, np.log(max(P @ heat, 1e-300)), (P @ cool + auto) / (x * np.sqrt(T))] + lnD + lnalpha
    return np.array(vals), r["Zstar"]


def _row(args):
    tab, lt, lx = args
    gt = rg.GroupChargingTable(tab, kernels="pchip")
    out = np.zeros((len(QUANTITIES), lx.size))
    zg = None
    for j, l in enumerate(lx):                          # warm start along x
        out[:, j], zg = _node(gt, 10.0 ** lt, 10.0 ** l, zg)
    return out


def build_isrf_table(tab, workers=1, lT=LT, lX=LX):
    """The table of one dust bin (a ``read_bin_table`` dict)."""
    args = [(tab, lt, np.asarray(lX, float)) for lt in lT]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_row, args))
    else:
        rows = [_row(a) for a in args]
    return dict(version=FORMAT_VERSION, material=tab["material"], a=tab["a"],
                lT=np.asarray(lT, float), lX=np.asarray(lX, float), q=np.stack(rows, axis=1))   # (nq, nT, nX)


# --------------------------------------------------------------------------- file
# text: header lines starting with '#', then
#   version  nT  nX  nQ  a [cm]      (RAMSES checks a against the bin's grain radius)
#   log10 T nodes (nT)
#   log10 x nodes (nX)
#   the nQ quantities in QUANTITIES order, each nT rows of nX values
def write_isrf_table(t, path, label=""):
    path = Path(path)
    with open(path, "w") as f:
        f.write("# pyCALIMA WDB06 grain charging under a uniform ISRF (Mathis, G0 = 1 Habing), "
                f"format {t['version']}\n")
        f.write(f"# bin {label}: {t['material']}, a = {t['a']:.6e} cm; ions: H+ with n = n_e; x = n_e/G0 [cm^-3]\n")
        f.write("# quantities (each nT x nX, T slow): " + " ".join(QUANTITIES) + "\n")
        f.write("# Gamma = G0 exp(lnGamma1), Lambda = n_e sqrt(T) lambda1 [erg/s per grain]; "
                "D_j = exp(lnD_j); alpha_i = exp(lnalpha_i) [cm^3/s per grain]; "
                "see pycalima.models.dust_charge.charging_isrf_tables\n")
        f.write(f"{t['version']} {t['lT'].size} {t['lX'].size} {len(QUANTITIES)} {t['a']:.8e}\n")
        f.write(" ".join(f"{v:.8e}" for v in t["lT"]) + "\n")
        f.write(" ".join(f"{v:.8e}" for v in t["lX"]) + "\n")
        for k in range(len(QUANTITIES)):
            for i in range(t["lT"].size):
                row = t["q"][k, i] if QUANTITIES[k] == "lambda1" or k < 2 else np.maximum(t["q"][k, i], LN_FLOOR)
                f.write(" ".join(f"{v:.10e}" for v in row) + "\n")
    return path


def read_isrf_table(path):
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.startswith("#")]
    head = lines[0].split()
    version, nT, nX, nQ = (int(v) for v in head[:4])
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: format {version}, expected {FORMAT_VERSION} (re-export the tables)")
    lT = np.array(lines[1].split(), float)
    lX = np.array(lines[2].split(), float)
    q = np.array([l.split() for l in lines[3:3 + nQ * nT]], float).reshape(nQ, nT, nX)
    return dict(version=version, a=float(head[4]), lT=lT, lX=lX, q=q)


def isrf_lookup(t, G0, T, ne):
    """<Z>, sigma_Z, Gamma, Lambda [erg/s per grain], D_j (D_CHARGES) and alpha_i (RECOMB_IONS)
    [cm^3/s per grain] of one bin's table at background G0 [Habing], T [K], n_e [cm^-3]:
    bilinear in (log T, log n_e/G0), clamped."""
    lt = min(max(np.log10(T), t["lT"][0]), t["lT"][-1])
    lx = min(max(np.log10(max(ne, 1e-300) / max(G0, 1e-300)), t["lX"][0]), t["lX"][-1])
    fi = (lt - t["lT"][0]) / (t["lT"][1] - t["lT"][0])
    fj = (lx - t["lX"][0]) / (t["lX"][1] - t["lX"][0])
    i = min(int(fi), t["lT"].size - 2)
    j = min(int(fj), t["lX"].size - 2)
    u, v = fi - i, fj - j
    q = t["q"]
    c = (1 - u) * (1 - v) * q[:, i, j] + u * (1 - v) * q[:, i + 1, j] + (1 - u) * v * q[:, i, j + 1] + u * v * q[:, i + 1, j + 1]
    nD = len(D_CHARGES)
    return dict(Zmean=c[0], Zsigma=c[1], Gamma=G0 * np.exp(c[2]), Lambda=ne * np.sqrt(T) * c[3],
                D=np.exp(c[4:4 + nD]), alpha=np.exp(c[4 + nD:]))
