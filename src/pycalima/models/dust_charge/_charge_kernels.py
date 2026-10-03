"""Compiled core of the per-cell charge solve (GroupChargingTable): g(Z) and its root.

The Python solver spends almost all its time in g(Z) = ln Up(Z) - ln Down(Z+1), called with
one charge at a time from the bracket and Illinois search. Here g and the search run in
numba on per-cell arrays (cell_arrays): the photon-weighted kernel sum_g cN_g K_g(U) and its
PCHIP slopes, summed once per cell (the interpolation is linear in them), and the hot tables
(delta, P_tr) interpolated once at the cell's temperature. Without numba, NUMBA is False and
GroupChargingTable keeps its Python path.
"""
import numpy as np

from pycalima.models.dust_charge.shared_physics import E_STATC, EV2ERG, KB_CGS, _ds87_j_scalar_impl

try:
    from numba import njit
    NUMBA = True
except Exception:  # pragma: no cover - optional acceleration
    NUMBA = False

U_SCALE = 0.5
TINY_RATE = 1e-300


def _kern(y, m, pchip, i, w):
    if not pchip:
        return (1 - w) * y[i] + w * y[i + 1]
    w2 = w * w
    w3 = w2 * w
    return ((2 * w3 - 3 * w2 + 1) * y[i] + (w3 - 2 * w2 + w) * m[i]
            + (3 * w2 - 2 * w3) * y[i + 1] + (w3 - w2) * m[i + 1])


def _u_index(U, x0, dx, nU):
    f = (np.arcsinh(U / U_SCALE) - x0) / dx
    f = min(max(f, 0.0), nU - 1.0)
    i = min(int(f), nU - 2)
    return i, f - i


def _arrive(Z, q, a, T, n, v):
    return np.pi * a * a * n * v * _ds87_j_scalar_impl(Z, q, a, T)


def g_scalar(Z, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T, ion_n, ion_z, ion_v):
    """ln Up(Z) - ln Down(Z + 1) of the cell (GroupChargingTable._g)."""
    nU = Kc.size
    i, w = _u_index(Z * e2a, x0, dx, nU)
    photo = _kern(Kc, mKc, pchip, i, w)
    if pchip:
        photo = max(photo, 0.0)
    up = photo + _arrive(Z, -1.0, a, T, ne, v_e) * ((1 - w) * delta[i] + w * delta[i + 1])
    for k in range(ion_n.size):
        up += _arrive(Z, ion_z[k], a, T, ion_n[k], ion_v[k])
    Z1 = Z + 1.0
    i, w = _u_index(Z1 * e2a, x0, dx, nU)
    stick = 0.0 if Z1 <= Zmin else (base if Z1 > 0 else neg)
    down = _arrive(Z1, -1.0, a, T, ne, v_e) * stick * (1.0 - ((1 - w) * Ptr[i] + w * Ptr[i + 1]))
    return np.log(max(up, TINY_RATE)) - np.log(max(down, TINY_RATE))


def g_array(Z, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T, ion_n, ion_z, ion_v):
    out = np.empty(Z.size)
    for k in range(Z.size):
        out[k] = g_scalar(Z[k], Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T,
                          ion_n, ion_z, ion_v)
    return out


def illinois(a_, b_, fa, fb, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T,
             ion_n, ion_z, ion_v):
    """Root of g in [a_, b_] with g(a_) > 0 > g(b_) (GroupChargingTable._illinois)."""
    side = 0
    Zs = b_
    for _ in range(200):
        Zs = (a_ * fb - b_ * fa) / (fb - fa)
        if abs(b_ - a_) < 1e-7 * max(1.0, abs(Zs)):
            break
        fz = g_scalar(Zs, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T,
                      ion_n, ion_z, ion_v)
        if fz == 0:
            break
        if fz * fb > 0:
            b_, fb = Zs, fz
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a_, fa = Zs, fz
            if side == 1:
                fb *= 0.5
            side = 1
    return Zs


def root_from_guess(Zg, Zlo, Zhi, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T,
                    ion_n, ion_z, ion_v):
    """Bracket the root by doubling steps from Zg, then Illinois (GroupChargingTable._root_from_guess)."""
    fg = g_scalar(Zg, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T, ion_n, ion_z, ion_v)
    if fg == 0:
        return Zg
    step = 1.0
    if fg > 0:
        a_, fa = Zg, fg
        while True:
            b_ = min(a_ + step, Zhi)
            fb = g_scalar(b_, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T,
                          ion_n, ion_z, ion_v)
            if fb == 0:
                return b_
            if fb < 0:
                return illinois(a_, b_, fa, fb, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg,
                                ne, v_e, T, ion_n, ion_z, ion_v)
            if b_ == Zhi:
                return Zhi
            a_, fa = b_, fb
            step *= 2.0
    b_, fb = Zg, fg
    while True:
        a_ = max(b_ - step, Zlo)
        fa = g_scalar(a_, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T,
                      ion_n, ion_z, ion_v)
        if fa == 0:
            return a_
        if fa > 0:
            return illinois(a_, b_, fa, fb, Kc, mKc, pchip, delta, Ptr, x0, dx, e2a, a, Zmin, base, neg,
                            ne, v_e, T, ion_n, ion_z, ion_v)
        if a_ == Zlo:
            return Zlo
        b_, fb = a_, fa
        step *= 2.0


def _lambda_electron(Z, a, T):
    """DS87 Lambda~ of electrons at grain charge Z (rt_group_charging._ds87_lambda_electron_arr)."""
    nu = -Z
    tau = max(a * KB_CGS * T / E_STATC ** 2, 1e-300)
    if nu == 0.0:
        return 2.0 + 1.5 * np.sqrt(np.pi / (2.0 * tau))
    if nu < 0.0:
        return (2.0 - nu / tau) * (1.0 + 1.0 / np.sqrt(max(tau - nu, 1e-300)))
    theta = nu / (1.0 + 1.0 / np.sqrt(nu))
    return (2.0 + nu / tau) * (1.0 + 1.0 / np.sqrt(1.5 / tau + 3.0 * nu)) * np.exp(-theta / tau)


def heating_cooling_array(Z, Hc, mHc, pchip, delta, fcool, Ebar, x0, dx, e2a, a, Zmin, base, neg, ne, v_e, T):
    """Photoelectric heating and recombination cooling [erg/s per grain] at the charges Z
    (GroupChargingTable._heating_cooling_arr)."""
    gam = np.empty(Z.size)
    lam = np.empty(Z.size)
    arrive = np.pi * a * a * ne * v_e
    for k in range(Z.size):
        z = Z[k]
        i, w = _u_index(z * e2a, x0, dx, Hc.size)
        h = _kern(Hc, mHc, pchip, i, w)
        gam[k] = EV2ERG * (max(h, 0.0) if pchip else h)
        stick = 0.0 if z <= Zmin else (base if z > 0 else neg)
        lam[k] = (arrive * stick * _lambda_electron(z, a, T) * KB_CGS * T * ((1 - w) * fcool[i] + w * fcool[i + 1])
                  - arrive * _ds87_j_scalar_impl(z, -1.0, a, T) * ((1 - w) * delta[i] + w * delta[i + 1])
                  * ((1 - w) * Ebar[i] + w * Ebar[i + 1]) * EV2ERG)
    return gam, lam


if NUMBA:
    _lambda_electron = njit(cache=True)(_lambda_electron)
    heating_cooling_array = njit(cache=True)(heating_cooling_array)
    _kern = njit(cache=True)(_kern)
    _u_index = njit(cache=True)(_u_index)
    _arrive = njit(cache=True)(_arrive)
    g_scalar = njit(cache=True)(g_scalar)
    g_array = njit(cache=True)(g_array)
    illinois = njit(cache=True)(illinois)
    root_from_guess = njit(cache=True)(root_from_guess)


def _at_T(tab, lt, T):
    y = min(max(np.log10(T), lt[0]), lt[-1])
    fT = (y - lt[0]) / ((lt[-1] - lt[0]) / (lt.size - 1))
    k = min(int(fT), lt.size - 2)
    v = fT - k
    return np.ascontiguousarray((1 - v) * tab[k] + v * tab[k + 1], float)


def heat_arrays(gt):
    """The per-cell arguments of heating_cooling_array after _set_cell (a tuple)."""
    t, c = gt.t, gt.cell
    pchip = gt.kernels == "pchip"
    Hc = c["cN"] @ gt.Hg + c["G0_bg"] * t["H_bg"]
    mHc = (c["cN"] @ gt._mH + c["G0_bg"] * gt._mHb) if pchip else np.zeros(1)
    lt, T = t["log10_Thot"], c["T"]
    a = float(gt.a)
    base = 0.5 * (1.0 - np.exp(-a / 1e-7))
    neg = base / (1.0 + np.exp(20.0 - 468.0 * (a / 1e-7) ** 3))
    return (np.ascontiguousarray(Hc, float), np.ascontiguousarray(mHc, float), pchip, _at_T(t["delta"], lt, T),
            _at_T(t["f_cool"], lt, T), _at_T(t["E_bar"], lt, T), float(gt.x0), float(gt.dx), float(gt.e2a), a,
            float(t["Zmin"]), float(base), float(neg), float(c["ne"]), float(c["v_e"]), float(T))


def cell_arrays(gt):
    """The per-cell arguments of g for GroupChargingTable gt after _set_cell (a tuple)."""
    t, c = gt.t, gt.cell
    pchip = gt.kernels == "pchip"
    Kc = c["cN"] @ gt.Kg + c["G0_bg"] * t["K_bg"]
    mKc = (c["cN"] @ gt._mK + c["G0_bg"] * gt._mKb) if pchip else np.zeros(1)
    lt = t["log10_Thot"]
    y = min(max(np.log10(c["T"]), lt[0]), lt[-1])
    fT = (y - lt[0]) / ((lt[-1] - lt[0]) / (lt.size - 1))
    k = min(int(fT), lt.size - 2)
    v = fT - k
    delta = (1 - v) * t["delta"][k] + v * t["delta"][k + 1]
    Ptr = (1 - v) * t["P_tr"][k] + v * t["P_tr"][k + 1]
    a = float(gt.a)
    base = 0.5 * (1.0 - np.exp(-a / 1e-7))
    neg = base / (1.0 + np.exp(20.0 - 468.0 * (a / 1e-7) ** 3))
    ions = c["ions"]
    return (np.ascontiguousarray(Kc, float), np.ascontiguousarray(mKc, float), pchip,
            np.ascontiguousarray(delta, float), np.ascontiguousarray(Ptr, float), float(gt.x0), float(gt.dx),
            float(gt.e2a), a, float(t["Zmin"]), float(base), float(neg), float(c["ne"]), float(c["v_e"]),
            float(c["T"]), np.array([n for n, _, _ in ions], float), np.array([z for _, z, _ in ions], float),
            np.array([v_ for _, _, v_ in ions], float))
