"""Tabulate or track the hydrogen of PAHs? Timescales, fast charge, a one-scalar closure, parcels.

The PAH half of test P2 of the carbon-carrier plan (the a-C(:H) half is ``gsd.hydrogen``). A PAH family is a (Z, N_H)
``network.PAHNetwork`` (``baseline.BaselineModel.network``, ``montillaud13.Montillaud13.network``, ...); these tools
ask of it what a RAMSES bin that stores only M_C and M_H needs to know:

- ``relaxation_rates``: the decay rates of the network's modes (|Re| of the non-zero eigenvalues of its generator),
  and ``charge_rates``: those of its charge steps alone, at each N_H.
- ``settling_time``: when an observable has made a share ``frac`` of its change after a step in the conditions.
- ``eliminate_charge`` (test T2, fast charge): the N_H chain whose rates out of each N_H are averaged over the
  charge distribution of that N_H (``conditional=True``, exact when charge is much faster than hydrogen), or over
  that of the intact molecule for every N_H (``conditional=False``: what ``population`` does, a charge table with no
  N_H axis); ``lift`` turns its populations back into f(Z, N_H).
- ``Closure`` (test T3): one tracked scalar, <N_H>. The distribution is rebuilt from it as the steady state of the
  same network with every process that adds hydrogen scaled by one factor lambda (lambda = 1: the true steady
  state), and <N_H> moves at R = sum over processes of dN_H x flux in that distribution. Exact in steady state, an
  assumption out of it.
- ``evolve_piecewise``: populations along a path of conditions held over segments (tests T4, T6).
- ``observables``: what the tests compare (fractions of the hydrogenation and charge states, H2 formation, and the
  aliphatic / aromatic CH ratio of the states, a proxy for 3.4 / 3.3 um that weights every bond and state alike).

Nothing here is physics: the rates are the network's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from pycalima.pahchem.network import PAHNetwork, gth_stationary

YR = 3.15576e7
OBSERVABLES = ("x", "f_intact", "f_superh", "f_bare", "f_cation", "z_mean", "r_ali", "h2")


def relaxation_rates(net, n=3):
    """The ``n`` slowest decay rates [s^-1] of ``net``, slowest first: |Re lambda| of the eigenvalues of its
    generator, without the stationary one (the eigenvalue nearest 0)."""
    ev = np.abs(np.linalg.eigvals(net.generator().toarray()).real)
    ev = np.delete(ev, np.argmin(ev))
    return np.sort(ev)[:n]


def _charge_block(net, j):
    """Generator rows of the charge steps (dN_H = 0) at N_H index j: R[a, b] = rate of Z_a -> Z_b."""
    nz = len(net.charges)
    R = np.zeros((nz, nz))
    for p in net.processes:
        if p.dnh == 0 and p.dz != 0:
            for a in range(nz):
                if p.rate[a, j] > 0:
                    R[a, a + p.dz] += p.rate[a, j]
    return R


def charge_rates(net):
    """The slowest charge-relaxation rate [s^-1] at each N_H of ``net`` (its charge steps alone, N_H held)."""
    out = np.zeros(len(net.nh_values))
    for j in range(out.size):
        R = _charge_block(net, j)
        Q = R.T - np.diag(R.sum(axis=1))
        ev = np.abs(np.linalg.eigvals(Q).real)
        out[j] = np.sort(ev)[1] if ev.size > 1 else np.inf
    return out


def charge_given_nh(net):
    """f(Z | N_H) [n_Z, n_NH]: the steady state of the charge steps alone at each N_H."""
    fz = np.zeros(net.shape)
    for j in range(fz.shape[1]):
        p = gth_stationary(_charge_block(net, j))
        if p is None:
            raise ValueError(f"the charge chain at N_H = {net.nh_values[j]} has no unique steady state")
        fz[:, j] = p
    return fz


def eliminate_charge(net, conditional=True, nh0=None):
    """(chain, f(Z | N_H)): the N_H-only network of fast charge, and the charge distribution it assumes.

    Every process that changes N_H keeps its dN_H (and its H2), at the rate sum_Z f(Z | N_H) rate(Z, N_H). With
    ``conditional=False``, f(Z | N_H) is that of N_H0 (``nh0``) for every N_H."""
    fz = charge_given_nh(net)
    if not conditional:
        fz = np.repeat(fz[:, [net.nh_values.index(nh0)]], fz.shape[1], axis=1)
    chain = PAHNetwork((0,), net.nh_values)
    for p in net.processes:
        if p.dnh != 0:
            chain.add(p.name, (fz * p.rate).sum(axis=0)[None, :], dz=0, dnh=p.dnh, h2_released=p.h2_released)
    return chain, fz


def lift(p_nh, fz):
    """f(Z, N_H) = f(Z | N_H) p(N_H), from chain populations ``p_nh`` [..., 1, n_NH] or [..., n_NH]."""
    p = np.asarray(p_nh)
    p = p.reshape(p.shape[:-2] + (p.shape[-1],)) if p.ndim >= 2 and p.shape[-2] == 1 else p
    return p[..., None, :] * fz


def observables(net, f, nh0):
    """{name: value} of populations f[Z, N_H] (or a stack [..., Z, N_H]) of ``net`` (``OBSERVABLES``):

    - x: <N_H> / N_H0; f_intact, f_superh, f_bare: shares at N_H = N_H0, > N_H0, = 0;
    - f_cation: share at Z > 0; z_mean: <Z>;
    - r_ali: aliphatic over aromatic CH, each extra H making a CH2 out of an aromatic CH (2 aliphatic CH, one
      aromatic CH less): a 3.4/3.3 um proxy with one band strength per bond and one emission weight per state;
    - h2: H2 released per PAH per s."""
    f = np.asarray(f, float)
    nh = np.array(net.nh_values)
    z = np.array(net.charges)
    p = f.sum(axis=-2)
    fz = f.sum(axis=-1)
    extra = np.maximum(nh - nh0, 0)
    aro = np.minimum(nh, nh0) - extra
    h2 = sum(q.h2_released * (f * q.rate).sum(axis=(-2, -1)) for q in net.processes)
    return {"x": p @ nh / nh0, "f_intact": p[..., nh == nh0].sum(axis=-1), "f_superh": p[..., nh > nh0].sum(axis=-1),
            "f_bare": p[..., 0], "f_cation": fz[..., z > 0].sum(axis=-1), "z_mean": fz @ z,
            "r_ali": (p @ (2 * extra)) / np.maximum(p @ aro, 1e-300), "h2": np.asarray(h2, float)}


def evolve(net, f0, times):
    """f at ``times`` [s] from f0 under fixed conditions (dense expm, as ``PAHNetwork.evolve``)."""
    return net.evolve(f0, times)


def evolve_piecewise(nets, durations, f0):
    """f after each segment: ``nets[k]`` held for ``durations[k]`` [s], from f0. Returns [n_segments + 1, Z, N_H]."""
    f = np.asarray(f0, float).ravel()
    out = [f.reshape(nets[0].shape).copy()]
    for net, dt in zip(nets, durations):
        f = scipy.linalg.expm(net.generator().toarray() * dt) @ f
        f = np.clip(f, 0.0, None)
        f /= f.sum()
        out.append(f.reshape(net.shape).copy())
    return np.array(out)


def settling_time(times, y, frac=0.9, y_inf=None):
    """The first time after which y stays within (1 - frac) of its total change from y[0] to y_inf (default the last
    value); 0 if y never leaves that band, inf if it is still outside it at the last time."""
    y = np.asarray(y, float)
    y_inf = y[-1] if y_inf is None else y_inf
    span = abs(y[0] - y_inf)
    if span == 0.0:
        return 0.0
    out = np.nonzero(np.abs(y - y_inf) > (1.0 - frac) * span)[0]
    if out.size == 0:
        return 0.0
    return np.inf if out[-1] + 1 >= len(times) else float(times[out[-1] + 1])


def population_chains(pop, env):
    """[PAHNetwork] of the hydrogenation chain of every molecule of ``pop`` (a ``population.Population``) at ``env``.

    The chain is the population's own (``Population._chain_distributions``): N_H = 0 ... N_H0 at Z = 0 only, its
    H and H2 loss and H addition averaged over the intact molecule's charge fractions, with the deposits, the
    protons past a barrier-limited ladder top and the hot ions' knock-outs. Its steady state is the population's
    (tests/test_pahchem_hydrogen_tracking.py). It has no super-hydrogenated states."""
    from pycalima.pahchem import population as P

    f = pop.charge_matrix(env)
    tab_h, tab_h2, _, tab_hp = pop._ladders(P.field_shape(env))
    lh = np.einsum("sz,szk->sk", f, pop._interp_g0(tab_h, env.g0) + pop._interp_g0(tab_hp, env.g0))
    lh2 = np.einsum("sz,szk->sk", f, pop._interp_g0(tab_h2, env.g0))
    dlh, dlh2, _, dlhp = pop._deposit_losses(env, f)
    lh, lh2 = lh + dlh + dlhp, lh2 + dlh2
    k_add = np.einsum("sz,sz->s", f, pop._addition_rates(env))
    routed = pop._routed()
    per_mol = (np.where(routed, (f * pop._past_top(env, pop._single_up(env))[1]).sum(axis=1), 0.0)
               if routed.any() else np.zeros(len(pop.library)))
    hot = pop._hot_rates(env)
    per_h = (f * hot["knockout_h"]).sum(axis=1) if hot is not None else np.zeros(len(pop.library))
    out = []
    for s, mol in enumerate(pop.library):
        kH, kH2 = P._chain_rates(mol.nh, mol.edges.get("solo", 0), lh[s], lh2[s], per_mol[s], per_h[s])
        nh = np.arange(mol.nh + 1)
        net = PAHNetwork((0,), nh)
        net.add("H loss", kH[None, :], dnh=-1)
        net.add("H2 loss", kH2[None, :], dnh=-2, h2_released=1)
        net.add("H addition", np.where(nh < mol.nh, k_add[s], 0.0)[None, :], dnh=+1)
        out.append(net)
    return out


def _scaled(net, lam):
    """``net`` with every process that adds hydrogen (dN_H > 0) times lam."""
    out = PAHNetwork(net.charges, net.nh_values)
    for p in net.processes:
        out.processes.append(type(p)(p.name, p.rate * (lam if p.dnh > 0 else 1.0), p.dz, p.dnh, p.h2_released))
    return out


@dataclass
class Closure:
    """The one-scalar closure of ``net`` (module docstring), tabulated on ``log_lam`` (ln lambda).

    ``x`` [n_lam] is <N_H> / N_H0 of each rebuilt distribution ``f`` [n_lam, Z, N_H], ``rate`` [n_lam] its
    d x / dt [s^-1] under the true rates. The rebuilt distributions span x from the hydrogen-poorest to the
    hydrogen-richest that the grid of lambda reaches; outside it they are clipped."""
    net: PAHNetwork
    nh0: int
    log_lam: np.ndarray = None

    def __post_init__(self):
        if self.log_lam is None:
            self.log_lam = np.linspace(-30.0, 30.0, 121)
        nh = np.array(self.net.nh_values)
        x, f = [], []
        for ll in self.log_lam:
            try:
                g = _scaled(self.net, np.exp(ll)).steady_state()
            except ValueError:              # no unique steady state (e.g. no H loss at all): not in the family
                continue
            xx = float(g.sum(axis=0) @ nh / self.nh0)
            if not x or xx > x[-1] + 1e-9:  # x rises with lambda; the flat ends add nothing
                x.append(xx)
                f.append(g)
        if len(x) < 2:
            raise ValueError("the closure family spans no range of <N_H>")
        self.x, self.f = np.array(x), np.array(f)
        flux = np.zeros(len(self.x))
        for p in self.net.processes:
            if p.dnh != 0:
                flux += p.dnh * (self.f * p.rate).sum(axis=(1, 2))
        self.rate = flux / self.nh0
        self.x_eq = float(observables(self.net, self.net.steady_state(), self.nh0)["x"])

    def distribution(self, x):
        """The rebuilt f[Z, N_H] at x (linear in x between the tabulated distributions)."""
        x = float(np.clip(x, self.x[0], self.x[-1]))
        i = int(np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2))
        w = (x - self.x[i]) / (self.x[i + 1] - self.x[i])
        return (1.0 - w) * self.f[i] + w * self.f[i + 1]

    def dxdt(self, x):
        return float(np.interp(x, self.x, self.rate))

    def linear_rate(self):
        """-dR/dx at the steady state [s^-1]: the rate of the linearised closure."""
        i = int(np.clip(np.searchsorted(self.x, self.x_eq), 1, len(self.x) - 1))
        lo, hi = max(i - 2, 0), min(i + 2, len(self.x) - 1)
        return float(-np.polyfit(self.x[lo:hi + 1], self.rate[lo:hi + 1], 1)[0])

    def secant_rate(self, x):
        """-R(x) / (x - x_eq) [s^-1]: the relaxation rate at x (the linearised rate at x_eq)."""
        if abs(x - self.x_eq) < 1e-9:
            return self.linear_rate()
        return max(-self.dxdt(x) / (x - self.x_eq), 0.0)

    def exp_step(self, x, dt):
        """x after dt with the rate frozen at x: x_eq + (x - x_eq) exp(-k(x) dt), k the secant rate (what RAMSES would
        apply, as ``gsd.hydrogen.BinHydrogen.step``)."""
        return self.x_eq + (x - self.x_eq) * np.exp(-self.secant_rate(x) * dt)

    def step(self, x, dt, n_sub=None):
        """x after dt [s] under these conditions: the closure's ODE, integrated exactly in x (dt = int dx / R)."""
        from scipy.integrate import solve_ivp

        if dt <= 0.0 or abs(self.dxdt(x)) == 0.0:
            return x
        sol = solve_ivp(lambda t, y: [self.dxdt(y[0])], (0.0, dt), [x], method="LSODA", rtol=1e-8,
                        atol=1e-10 * max(1.0, abs(x)))
        y = float(sol.y[0, -1])
        lo, hi = sorted((x, self.x_eq))
        return float(np.clip(y, lo, hi))           # the 1-D flow cannot cross its fixed point
