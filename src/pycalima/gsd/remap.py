"""Size-space transport: flow maps, conservative remap and nucleation injection.

Growth and sublimation move every grain along a characteristic
da/dt = adot(a), so a grain at a ends a step at X(a). ``remap`` takes the
two-moment power-law profile of each source bin (``pl2m``), splits it into
sub-intervals whose images are interpolated linearly in a (exact when adot
is constant), cuts the images at the fixed bin edges and integrates number
and mass of each piece analytically. Number moves along characteristics
exactly; mass is the grains' new mass. Characteristics never cross, so there
is no CFL limit. Grains whose image falls below the smallest edge are
removed (their mass goes back to the gas through the caller's closure),
grains above the largest edge are reported as overflow.

``inject`` lays down a nucleation source of rate J [cm^-3 s^-1] over a step:
grains born at the critical radius a* at time tau into the step end where
the characteristic from a* takes them in the remaining time, so the number
is spread with density J along that trajectory (NK13's integral over
formation time, binned).
"""

from __future__ import annotations

import numpy as np

from .pl2m import BinProfile, int_pow


def flow_map(adot, a, dt, rtol=1e-9, a_floor=0.0):
    """Positions after dt of grains at radii a under da/dt = adot(a).

    All characteristics are integrated together with an adaptive RK45
    (``rtol``); a grain reaching ``a_floor`` stays there (it has vanished).
    """
    from scipy.integrate import solve_ivp

    a = np.asarray(a, dtype=float)
    if dt == 0.0 or a.size == 0:
        return a.copy()
    scale = max(float(np.max(np.abs(a))), 1e-30)

    def rhs(_, x):
        x = np.maximum(x, a_floor)
        return np.where(x > a_floor, adot(x), np.maximum(adot(x), 0.0))

    sol = solve_ivp(rhs, (0.0, dt), a.ravel(), method="RK45", rtol=rtol,
                    atol=1e-12 * scale)
    return np.maximum(sol.y[:, -1], a_floor).reshape(a.shape)


def trajectory(adot, a0, dt, n_nodes=33, rtol=1e-9, a_floor=0.0):
    """Radius along the characteristic from a0 at times 0..dt (n_nodes)."""
    from scipy.integrate import solve_ivp

    s = np.linspace(0.0, dt, n_nodes)

    def rhs(_, x):
        x = np.maximum(x, a_floor)
        return np.where(x > a_floor, adot(x), np.maximum(adot(x), 0.0))

    sol = solve_ivp(rhs, (0.0, dt), [a0], method="RK45", rtol=rtol,
                    atol=1e-12 * max(abs(a0), 1e-30), t_eval=s)
    return s, np.maximum(sol.y[0], a_floor)


def _cubic_moment(prof, i, c0, c1, u1, u2):
    """Integral of (c0 + c1 a)^3 dN over u in [u1, u2] in bin i (a = a_L u),
    i.e. the sum over grains of the cube of their linearly mapped radius."""
    aL = prof.aL[i]
    d0, d1 = c0, c1 * aL           # image radius = d0 + d1 u
    out = 0.0
    coef = (d0 ** 3, 3 * d0 ** 2 * d1, 3 * d0 * d1 ** 2, d1 ** 3)
    for k, ck in enumerate(coef):
        if ck != 0.0:
            out = out + ck * prof.moment(i, k, u1, u2)
    return out


def remap(grid, N, M, image, n_pieces=4):
    """Remap (N, M) per bin along the flow map ``image(a) -> X(a)``.

    ``image`` is a vectorised callable giving the new radius of grains now at
    radius a (e.g. ``lambda a: flow_map(adot, a, dt)``); it must be monotone
    non-decreasing. Returns (N_new, M_new, lost_N, overflow_N, overflow_M):
    the number of grains that vanished below the grid, and the number and
    mass carried above it.
    """
    prof = BinProfile(grid, N, M)
    nb = grid.n_bins
    a_e = grid.a_edges
    mfac = 4.0 / 3.0 * np.pi * grid.rho
    N_new = np.zeros(nb)
    M_new = np.zeros(nb)
    lost_N = over_N = over_M = 0.0
    src = np.nonzero(N > 0)[0]
    if src.size == 0:
        return N_new, M_new, 0.0, 0.0, 0.0
    # sub-interval nodes, geometric inside each bin, mapped in one call
    t = np.linspace(0.0, 1.0, n_pieces + 1)
    nodes = a_e[src, None] * (a_e[src + 1, None] / a_e[src, None]) ** t[None, :]
    X = image(nodes.ravel()).reshape(nodes.shape)
    for row, i in enumerate(src):
        aL = prof.aL[i]
        if prof.delta[i]:
            # a monodisperse bin moves as a whole along its characteristic
            Xd = float(image(np.array([aL * prof.u_delta[i]]))[0])
            j = np.searchsorted(a_e, Xd, side="right") - 1
            if j < 0 or Xd <= 0.0:
                lost_N += N[i]
            elif j >= nb:
                over_N += N[i]
                over_M += N[i] * mfac * Xd ** 3
            else:
                N_new[j] += N[i]
                M_new[j] += N[i] * mfac * Xd ** 3
            continue
        for k in range(n_pieces):
            p, q = nodes[row, k], nodes[row, k + 1]
            Xp, Xq = X[row, k], X[row, k + 1]
            up, uq = p / aL, q / aL
            if Xq - Xp <= 1e-14 * max(Xq, 1e-300):
                # the piece collapses onto one radius: keep it together
                n_piece = prof.number(i, up, uq)
                j = np.searchsorted(a_e, Xp, side="right") - 1
                if j < 0 or Xp <= 0.0:
                    lost_N += n_piece
                elif j >= nb:
                    over_N += n_piece
                    over_M += n_piece * mfac * Xp ** 3
                else:
                    N_new[j] += n_piece
                    M_new[j] += n_piece * mfac * Xp ** 3
                continue
            s = (Xq - Xp) / (q - p)
            c0, c1 = Xp - s * p, s
            # below the grid
            if Xp < a_e[0]:
                x_hi = min(Xq, a_e[0])
                u_hi = (p + (x_hi - Xp) / s) / aL
                lost_N += prof.number(i, up, u_hi)
            if Xq > a_e[-1]:
                x_lo = max(Xp, a_e[-1])
                u_lo = (p + (x_lo - Xp) / s) / aL
                over_N += prof.number(i, u_lo, uq)
                over_M += mfac * _cubic_moment(prof, i, c0, c1, u_lo, uq)
            j0 = max(np.searchsorted(a_e, Xp, side="right") - 1, 0)
            j1 = min(np.searchsorted(a_e, Xq, side="left") - 1, nb - 1)
            for j in range(j0, j1 + 1):
                x1 = max(Xp, a_e[j])
                x2 = min(Xq, a_e[j + 1])
                if x2 <= x1:
                    continue
                u1 = (p + (x1 - Xp) / s) / aL
                u2 = (p + (x2 - Xp) / s) / aL
                N_new[j] += prof.number(i, u1, u2)
                M_new[j] += mfac * _cubic_moment(prof, i, c0, c1, u1, u2)
    return N_new, M_new, lost_N, over_N, over_M


def inject(grid, J, positions):
    """Bin a nucleation source along a characteristic.

    ``positions`` are the end-of-step radii of grains born at evenly spaced
    times through the step (from ``trajectory`` read backwards, i.e. y[::-1]
    for a* followed forward), and ``J * dt`` [cm^-3] is the number formed in
    the step. Between successive positions the number is spread uniformly in
    radius; pieces below the grid are dropped (they never become grains).
    Returns (dN, dM) per bin.
    """
    nb = grid.n_bins
    a_e = grid.a_edges
    mfac = 4.0 / 3.0 * np.pi * grid.rho
    dN = np.zeros(nb)
    dM = np.zeros(nb)
    y = np.asarray(positions, float)
    n_seg = y.size - 1
    per_seg = J / n_seg
    for k in range(n_seg):
        lo, hi = sorted((y[k], y[k + 1]))
        if hi - lo <= 1e-14 * hi:
            j = np.searchsorted(a_e, lo, side="right") - 1
            if 0 <= j < nb:
                dN[j] += per_seg
                dM[j] += per_seg * mfac * lo ** 3
            continue
        dens = per_seg / (hi - lo)
        j0 = max(np.searchsorted(a_e, lo, side="right") - 1, 0)
        j1 = min(np.searchsorted(a_e, hi, side="left") - 1, nb - 1)
        for j in range(j0, j1 + 1):
            x1, x2 = max(lo, a_e[j]), min(hi, a_e[j + 1])
            if x2 > x1:
                dN[j] += dens * (x2 - x1)
                dM[j] += dens * mfac * (x2 ** 4 - x1 ** 4) / 4.0
    return dN, dM
