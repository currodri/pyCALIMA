"""The carbon pair rule: what molecules do in collisions (decided 2026-11-23; status report Sect. "How carbon moves").

Carbon lives on one grid (``carbon.CarbonLayout``); a grain's size decides its class:
- molecule: N_C up to N_C,max (400 by default, a <= 1.06 nm), the molecular bins' tables;
- nano-grain: above it, up to the grid edge at ~20 nm;
- grain: above that.

What happens in a collision depends on the pair:
- **molecule + molecule: no outcome.** van der Waals encounters make no covalent grains, and PAH clusters are not
  modelled.
- **molecule + larger particle, by impact speed** (labelled: C60 and cluster surface-impact data, scaled per atom; no
  PAH-specific measurements). The molecule
  - sticks below ``v_stick`` (~4 km/s, ~1 eV per atom),
  - rebounds intact between, which needs no channel,
  - is destroyed above ``v_destroy`` (~7 km/s, ~3 eV per atom, uncertain by a factor 2), its carbon going to the
    gas and the partner left intact.

  On nano-grains the sticking is multiplied by S_eff, the photo-release term (``release``; 1 unless the cell's
  operator sets it). The rate is sigma <v> over the speed distribution of the pair (``velocities.RelativeVelocity``),
  with the Coulomb factor of both partners' charges when the velocity model carries them.
- **grain + grain:** the grain collision physics (``shattering.GrainCollisions``), unchanged.

``GrainCollisions(pair_rule=PairRule(...))`` applies the rule.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .collisions import Channel
from .velocities import DV_MIN, coulomb_rate_factor, log_flux_above

_E2 = 4.80320471e-10 ** 2
#: carbon's class boundary between nano-grains and grains [cm], placed on the grid edge nearest to it
A_NANO_CM = 20e-7
#: impact speeds [cm/s] of the rule (labelled)
V_STICK, V_DESTROY = 4.0e5, 7.0e5


@dataclass
class PairRule:
    """The rule for one carbon grid: molecules are grains with a <= ``a_mol``, nano-grains those up to ``a_nano``
    [cm]. ``release(a_molecule, a_host)`` (vectorised, broadcasting) returns S_eff for a molecule sticking to a
    nano-grain; None means 1."""

    a_mol: float
    a_nano: float
    v_stick: float = V_STICK
    v_destroy: float = V_DESTROY
    release: Callable | None = None

    @classmethod
    def from_layout(cls, layout, grid=None, **kw) -> PairRule:
        """The class boundaries on ``layout``'s grid: N_C,max, and the grid edge nearest 20 nm."""
        g = grid or layout.grid()
        a_e = g.a_edges
        a_mol = float(a_e[layout.n_pah_bins])
        a_nano = float(a_e[int(np.argmin(np.abs(np.log(a_e / A_NANO_CM))))])
        if not a_nano > a_mol:
            raise ValueError("the nano-grain range must lie above the molecular one")
        return cls(a_mol, a_nano, **kw)

    def classes(self, a):
        """(molecule, nano-grain) masks of the radii ``a``. The class is a property of the grain's size: a grain
        exactly at a boundary takes the smaller class."""
        a = np.asarray(a, float)
        mol = a <= self.a_mol * (1.0 + 1e-9)
        return mol, ~mol & (a <= self.a_nano * (1.0 + 1e-9))

    def apply(self, chans, a, m, velocity, rho, a_frag_min):
        """The channels ``chans`` of the grain physics at node radii ``a`` and masses ``m``, with the rule applied:
        every channel zeroed for pairs with a molecule, plus a stick and a destroy channel for molecule + larger
        particle. ``rho`` is the bulk density of the velocity model's stopping times; ``a_frag_min`` [cm] the grid's
        smallest radius (destroyed molecules go below it, into the collector)."""
        mol, nano = self.classes(a)
        mixed = mol[:, None] ^ mol[None, :]
        out = []
        for ch in chans:
            (r1, f1, h1, l1), (r2, f2, h2, l2) = ch.sets
            out.append(Channel(np.where(mixed | (mol[:, None] & mol[None, :]), 0.0, ch.K),
                               rem1=r1, frag1=f1, a_hi1=h1, a_lo1=l1, rem2=r2, frag2=f2, a_hi2=h2, a_lo2=l2))
        if not mixed.any():
            return out
        A1, A2 = a[:, None], a[None, :]
        M1, M2 = np.broadcast_to(m[:, None], mixed.shape), np.broadcast_to(m[None, :], mixed.shape)
        dV, mu = velocity.pair_statistics(a, m, rho)
        alpha = math.sqrt(1.5) / np.maximum(dV, DV_MIN)
        s = alpha * mu
        v_mean = velocity.collision_speeds(dV, mu, 0.0, 1)[0]
        f_stick, f_destroy = self._fractions(a, m, velocity, alpha, s)
        K = np.pi * (A1 + A2) ** 2 * v_mean
        S = np.ones(mixed.shape)
        if self.release is not None:
            host_row = nano[:, None] & mol[None, :]                   # rows nano-grain, columns molecule
            host_col = mol[:, None] & nano[None, :]
            S = np.where(host_col, self.release(A1, A2), S)
            S = np.where(host_row, self.release(A2, A1), S)
            S = np.clip(S, 0.0, 1.0)
        out.append(Channel(np.where(mixed, K * f_stick * S, 0.0), rem1=M1 + M2))
        tiny = np.full(mixed.shape, 0.5 * float(a_frag_min))
        r_mol, c_mol = np.broadcast_to(mol[:, None], mixed.shape), np.broadcast_to(mol[None, :], mixed.shape)
        out.append(Channel(np.where(mixed, K * f_destroy, 0.0),
                           rem1=np.where(r_mol, 0.0, M1), frag1=np.where(r_mol, M1, 0.0), a_hi1=tiny, a_lo1=tiny,
                           rem2=np.where(c_mol, 0.0, M2), frag2=np.where(c_mol, M2, 0.0), a_hi2=tiny, a_lo2=tiny))
        return out

    def _fractions(self, a, m, velocity, alpha, s):
        """Shares of the collision flux sigma <v> with impact speed below v_stick and above v_destroy, with the
        Coulomb factor summed over both partners' charge points if the velocity model has charges."""
        if velocity.charges is None:
            return (-np.expm1(log_flux_above(alpha * self.v_stick, s)),
                    np.exp(log_flux_above(alpha * self.v_destroy, s)))
        A1, A2 = a[:, None], a[None, :]
        M1, M2 = m[:, None], m[None, :]
        m_red = M1 * M2 / (M1 + M2)
        Z, wz = velocity.charges(a)
        f_stick = np.zeros(s.shape)
        f_destroy = np.zeros(s.shape)
        for i, wi in enumerate(wz):
            for j, wj in enumerate(wz):
                vC2 = 2.0 * _E2 * Z[:, i][:, None] * Z[:, j][None, :] / ((A1 + A2) * m_red)
                f_stick += wi * wj * coulomb_rate_factor(alpha, s, vC2, 0.0, self.v_stick)
                f_destroy += wi * wj * coulomb_rate_factor(alpha, s, vC2, self.v_destroy)
        return f_stick, f_destroy
