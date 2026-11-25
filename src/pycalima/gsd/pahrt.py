"""PAHrt: the PAH bins' equilibrium state per cell, from the local gas and the RT photon groups.

Decided 2026-11-24 (``diagnostics/pahchem/pahrt/README.md``): the molecular bins are in equilibrium, nothing about
them is tracked, and their state is solved in each cell from the cell's photon groups, T, n_e, n(H I) and ions. Each
bin is a few representative PAHdb molecules, each carrying its kernels from pahchem's joint (Z, N_H) chain
(``pahchem.pahrt_kernels``).

Per molecule (``MoleculeSolver``):

1. **Photon rates.** For every process and state, sum_g F_g kappa_g, with kappa_g interpolated between the bases'
   slopes at the group's mean photon energy. Multiphoton heating makes kappa depend on the field: by default group
   g's kappa is read at its own flux (log-log between pahchem's G0 nodes, extrapolated past them as pahchem does), so
   the photons of the other groups do not add to its multiphoton heating. Exact for single-photon processes and for a
   one-group field; at G0 1e2-1e5 on Draine and Kurucz fields, carbon loss within about +-0.3 dex of pahchem (bins
   1-2, 2026-11-25). Reading it where the group alone would give the cell's absorbed power or rate overstates the
   cross-group heating (``PROXIES``).
2. **Collision rates.** Per unit density on log T (linear in log T), times n_e, n(H I) and each ion.
3. **The steady state.** Either "full", the whole (Z, N_H) network (``network.PAHNetwork.steady_state_closed``), or
   "nested", RAMSES' algorithm with charge fast against hydrogenation (study S7). Nested takes the charge balance at
   each N_H, then the N_H ladder with charge-averaged rates.
4. **Outputs.** Sums over states of f times the per-state rates. The net electron source counts the photon- and
   collision-driven steps; ion-driven charge steps free no electron. alpha[ion] is the rate of the ion-driven steps
   per unit n(ion).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EV_ERG = 1.602176634e-12


@dataclass
class PAHrtCell:
    """The inputs of one cell. ``F``: group photon fluxes [cm^-2 s^-1] (c N_g plus the background's share); ``egy``:
    their mean photon energies [eV]; ``T`` [K]; ``n_e``, ``n_h`` (atomic H) [cm^-3]; ``ions``: ((name, n), ...)."""

    F: np.ndarray
    egy: np.ndarray
    T: float
    n_e: float
    n_h: float
    ions: tuple = ()


def electrons_per_event(name, dz):
    """Free electrons a photon- or collision-driven event makes (negative: captured)."""
    if name.startswith("ionisation"):
        return dz if "proton" not in name else dz + int(name.split("emitted")[0].split()[-2])
    if name == "associative detachment" or name.startswith("deposit dZ +1"):
        return 1
    if name.startswith("deposit dZ -1") or name.startswith("electron capture"):
        return -1
    return 0


def _loglog(x_nodes, y, x):
    """y[N0, ...] at x[...] (one x per trailing state), log-log between the nodes and extrapolated with the end
    intervals' slopes; linear (clamped) where a node pair holds a zero."""
    lx = np.log(x_nodes)
    lq = np.log(np.maximum(x, 1e-300))
    i = np.clip(np.searchsorted(lx, lq) - 1, 0, lx.size - 2)
    u = (lq - lx[i]) / (lx[i + 1] - lx[i])
    y0 = np.take_along_axis(y, i[None, None], axis=0)[0]
    y1 = np.take_along_axis(y, (i + 1)[None, None], axis=0)[0]
    pos = (y0 > 0) & (y1 > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ll = np.exp(np.log(np.where(pos, y0, 1.0)) * (1 - u) + np.log(np.where(pos, y1, 1.0)) * u)
    lin = y0 + np.clip(u, 0.0, 1.0) * (y1 - y0)
    return np.where(pos, ll, lin)


def _blend(y0, y1, t):
    """(1 - t) y0 + t y1 between two slopes (linear in the mean photon energy: a geometric blend doubles the error of
    threshold rates at a slope between the nodes, 2026-11-25)."""
    return y0 if t == 0.0 else (1 - t) * y0 + t * y1


class MoleculeSolver:
    """The per-cell state of one molecule from its ``pahrt_kernels.MoleculeKernels``."""

    #: the multiphoton proxy: group g's kappa is read at its own flux ("self": no heating by the other groups; the
    #: default, best against pahchem on Draine, Kurucz and old-star fields, 2026-11-25), or where its basis alone
    #: gives the cell's absorbed "power" or absorption "rate" (these overstate the cross-group heating)
    PROXIES = ("self", "power", "rate")

    def __init__(self, mk, proxy="self"):
        if proxy not in self.PROXIES:
            raise ValueError(f"proxy {proxy!r} not in {self.PROXIES}")
        self.proxy = proxy
        self.mk = mk
        self.z = np.asarray(mk.charges)
        self.nh = np.asarray(mk.nh_values)
        self.i_abs = mk.observables.index("absorbed")
        self.per_e = np.array([electrons_per_event(n, dz) for n, dz, *_ in mk.procs], float)
        self.h2 = np.array([h for *_, h in mk.procs], float)
        self.ions = {k: i for i, k in enumerate(mk.coll_kinds) if k not in ("e", "H")}
        self._k_e, self._k_h = mk.coll_kinds.index("e"), mk.coll_kinds.index("H")

    def _slopes(self, g, e):
        """(a0, a1, t): the bracketing slopes of group g at mean photon energy e (clamped to the bases')."""
        em = self.mk.e_mean[g]                              # falls with alpha
        order = np.argsort(em)
        x = em[order]
        if e <= x[0]:
            return order[0], order[0], 0.0
        if e >= x[-1]:
            return order[-1], order[-1], 0.0
        j = int(np.searchsorted(x, e)) - 1
        return order[j], order[j + 1], float((e - x[j]) / (x[j + 1] - x[j]))

    def photon(self, cell):
        """(K[P, Z, N], W[O, Z, N]): the photon-driven rates [s^-1] and observables."""
        mk = self.mk
        K = np.zeros(mk.photon_k.shape[3:])
        W = np.zeros(mk.photon_w.shape[3:])
        live = [g for g in np.nonzero(mk.valid)[0] if cell.F[g] > 0.0]
        if not live:
            return K, W
        sl = {g: self._slopes(g, cell.egy[g]) for g in live}
        p1 = {g: (1 - t) * mk.photon_w[g, a0, 0, self.i_abs] + t * mk.photon_w[g, a1, 0, self.i_abs]
              for g, (a0, a1, t) in sl.items()}
        x1 = p1 if self.proxy != "rate" else {g: p1[g] / cell.egy[g] for g in live}     # per unit F
        X = sum(cell.F[g] * x1[g] for g in live)
        for g in live:
            a0, a1, t = sl[g]
            if self.proxy == "self":
                fstar = np.full(p1[g].shape, cell.F[g])
            else:
                fstar = np.where(x1[g] > 0, X / np.where(x1[g] > 0, x1[g], 1.0), 0.0)
            pair = [(_loglog(mk.g0, mk.photon_k[g, a], fstar / mk.f_unit[g, a]),
                     _loglog(mk.g0, mk.photon_w[g, a], fstar / mk.f_unit[g, a])) for a in (a0, a1)]
            K += cell.F[g] * _blend(pair[0][0], pair[1][0], t)
            W += cell.F[g] * _blend(pair[0][1], pair[1][1], t)
        return K, W

    def collisions(self, cell):
        """(K_e + K_H [P, Z, N], {ion: (n, K_i per unit n [cm^3 s^-1])}, W [O, Z, N]): the collision-driven rates
        [s^-1] and observables."""
        mk = self.mk
        lt = np.log10(cell.T)
        j = int(np.clip(np.searchsorted(mk.lt, lt) - 1, 0, mk.lt.size - 2))
        u = float(np.clip((lt - mk.lt[j]) / (mk.lt[j + 1] - mk.lt[j]), 0.0, 1.0))
        ck = (1 - u) * mk.coll_k[:, j] + u * mk.coll_k[:, j + 1]
        cw = (1 - u) * mk.coll_w[:, j] + u * mk.coll_w[:, j + 1]
        K = cell.n_e * ck[self._k_e] + cell.n_h * ck[self._k_h]
        W = cell.n_e * cw[self._k_e] + cell.n_h * cw[self._k_h]
        Ki = {}
        for ion, n in cell.ions:
            if ion not in self.ions:
                raise ValueError(f"{mk.name}: no kernels for the ion {ion}")
            Ki[ion] = (n, ck[self.ions[ion]])
            W = W + n * cw[self.ions[ion]]
        return K, Ki, W

    def network(self, K):
        """A ``network.PAHNetwork`` with the rates K[P, Z, N]."""
        from pycalima.pahchem.network import PAHNetwork

        net = PAHNetwork(tuple(int(z) for z in self.z), tuple(int(n) for n in self.nh))
        for (name, dz, dnh, h2), r in zip(self.mk.procs, K, strict=True):
            if r.any():
                net.add(name, r, dz, dnh, h2)
        return net

    def nested(self, K):
        """F[Z, N]: the charge balance at each N_H, then the N_H ladder with charge-averaged rates. Every charge step
        out of a level enters its charge balance, also those that move N_H too (deposits, proton loss, associative
        detachment): the balance takes each state's whole up and down rates, the ladder the N_H moves."""
        from pycalima.pahchem.network import PAHNetwork

        nz, nn = K.shape[1:]
        fz = np.zeros((nz, nn))
        charge = [p for p, (_, dz, _, _) in enumerate(self.mk.procs) if dz != 0]
        ladder = [p for p, (_, _, dnh, _) in enumerate(self.mk.procs) if dnh != 0]
        zs = tuple(int(z) for z in self.z)
        for n in range(nn):
            net = PAHNetwork(zs, (0,))
            for p in charge:
                name, dz, _, _ = self.mk.procs[p]
                if K[p, :, n].any():
                    net.add(name, K[p, :, n][:, None], dz, 0)
            fz[:, n] = net.steady_state_closed()[:, 0]
        lad = PAHNetwork((0,), tuple(range(nn)))
        for p in ladder:
            name, _, dnh, h2 = self.mk.procs[p]
            r = (fz * K[p]).sum(axis=0)
            if r.any():
                lad.add(name, r[None, :], 0, dnh, h2)
        return fz * lad.steady_state_closed()[0][None, :]

    def solve(self, cell, method="nested"):
        """(F[Z, N], outputs) of the cell."""
        Kp, Wp = self.photon(cell)
        Kc, Ki, Wc = self.collisions(cell)
        K = Kp + Kc + sum((n * k for n, k in Ki.values()), np.zeros_like(Kp))
        F = self.nested(K) if method == "nested" else self.network(K).steady_state_closed()
        return F, self.outputs(F, Kp, Kc, Ki, Wp + Wc)

    def outputs(self, F, Kp, Kc, Ki, W):
        """Every output per molecule: f(Z), <Z>, the rates [erg s^-1, s^-1, cm^3 s^-1], the hydrogenation shares."""
        mk = self.mk
        fz = F.sum(axis=1)
        out = {f"f({int(z):+d})": float(f) for z, f in zip(self.z, fz, strict=True)}
        out["z_mean"] = float(fz @ self.z)
        out["z_width"] = float(np.sqrt(max(fz @ self.z ** 2 - out["z_mean"] ** 2, 0.0)))
        for j, k in enumerate(mk.observables):
            out[k] = float((W[j] * F).sum())
        flow = (Kp + Kc) * F[None]
        out["h2"] = float(self.h2 @ flow.sum(axis=(1, 2)))
        out["e_source"] = float(self.per_e @ flow.sum(axis=(1, 2)))
        for ion, (_, k) in Ki.items():
            out[f"alpha({ion})"] = float((k * F[None]).sum())
        fn = F.sum(axis=0)
        out["bare"] = float(fn[self.nh == 0].sum())
        out["dehydrogenated"] = float(fn[(self.nh > 0) & (self.nh < mk.nh0)].sum())
        out["intact"] = float(fn[self.nh == mk.nh0].sum())
        out["superh"] = float(fn[self.nh > mk.nh0].sum())
        out["dnh"] = float(fn @ (self.nh - mk.nh0))
        return out


#: atomic masses [g]
M_C_G, M_H_G = 12.011 * 1.66053906660e-24, 1.008 * 1.66053906660e-24


class PAHrtBin:
    """One molecular bin [nc_lo, nc_hi] as its representative molecules (``MoleculeSolver``s, at their N_C): every
    output is the number-weighted mean over the in-bin PL2M profile at the cell's s (``optics.profile_weights`` on the
    nodes u = (N_C / nc_lo)^(1/3), the bin's edges carrying their nearest molecule's values), exact in s for the
    molecules' values."""

    def __init__(self, solvers, nc_lo, nc_hi):
        self.solvers = sorted(solvers, key=lambda m: m.mk.nc)
        self.nc_lo, self.nc_hi = float(nc_lo), float(nc_hi)
        nc = np.array([m.mk.nc for m in self.solvers], float)
        self.mass = np.array([m.mk.nc * M_C_G + m.mk.nh0 * M_H_G for m in self.solvers])
        self.eta = self.nc_hi / self.nc_lo
        u = (nc / self.nc_lo) ** (1.0 / 3.0)
        self.u = np.concatenate([[1.0], u, [self.eta ** (1.0 / 3.0)]])
        self._fold = np.zeros((self.u.size, nc.size))           # node -> molecule (edges to the nearest)
        self._fold[0, 0] = self._fold[-1, -1] = 1.0
        self._fold[1:-1] = np.eye(nc.size)

    def weights(self, s):
        """Number weights of the molecules at in-bin state s (sum 1)."""
        from .optics import profile_weights

        W = profile_weights(self.eta, np.array([s]), self.u)[0] @ self._fold      # per-gram weights
        w = W / self.mass
        return w / w.sum()

    def solve(self, cell, s=0.5, method="nested"):
        """({output: number-weighted mean per molecule}, [per-molecule outputs], weights)."""
        outs = [m.solve(cell, method)[1] for m in self.solvers]
        w = self.weights(s)
        keys = sorted({k for o in outs for k in o})
        mean = {k: float(sum(wi * o.get(k, 0.0) for wi, o in zip(w, outs, strict=True))) for k in keys}
        return mean, outs, w
