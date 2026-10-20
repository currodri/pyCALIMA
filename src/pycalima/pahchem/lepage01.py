"""Le Page, Snow & Bierbaum (2001, 2003) run in their own configuration.

Inputs are those of method_cards/lepage01.toml (Paper I = ApJS 132, 233;
LP03 = ApJ 584, 316). Where the card says "not stated", the choice made here
is named in the docstring of the function that makes it, so a mismatch can
be traced to one entry:

- **Dissociation.** "2nu-RRKM" (Paper I Sect. 3.1.3): every C_xH_y has 3x - 6
  modes at 1000 cm^-1 and 3y at 1300 cm^-1. Its transition state drops one
  1300 cm^-1 mode and turns two more into bends at 250 cm^-1 (dS = 5 cal/K)
  or 950 cm^-1 (dS = 0). k(E) comes from ``rates.rrkm``, with the Table 3
  (E0, dS) of each channel.
- **Competition.** One photon only. Dissociation competes with the
  stepladder IR rate k'_rad of eq. 12 (q_IR = 0.18 eV, k_rad = 100 s^-1)
  at E = h nu; there is no cascade.
- **Field.** Draine (1978) above 5 eV, Mathis+83 below, scaled by chi.
- **Charge.** Ionisation by eq. 15 with the eq. 14 yield in um^-1; the
  benzene recombination rate x (300/T)^1/2, independent of size;
  attachment eq. 21; photodetachment eq. 22.
- **Hydrogenation network.** Z = 0, +1 and N_H = 0 .. 2y. Cation H
  addition follows Sect. 3.4, and recombination is dissociative by 2nu-RRKM
  branching at E = IP (Sect. 3.3).

Inputs read from figures, and substitutes:

- the Joblin soot cross section per C atom has no printed parameters; it
  is Paper I Fig. 11 as digitised (``SIGMA_FILE``). Li & Draine (2001) and
  Malloci coronene remain as options of ``sigma_uv``.
- No IPs are printed; Allain+96 I eq. 12 is used, the law of the paper Le
  Page cite for eq. 15.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import dos, fields, photo, rates
from pycalima.pahchem.network import PAHNetwork

EV = dos.CM_PER_EV
CAL = 4.184

NU_CC, NU_CH = 1000.0, 1300.0              # cm^-1 (Sect. 3.1.3)
TS_BEND = {5.0: 250.0, 0.0: 950.0}         # dS [cal/K/mol] -> TS bend frequency [cm^-1]
Q_IR_EV = 0.18                             # eV per IR photon (Sect. 3.1.5)
K_RAD = 100.0                              # s^-1, the energy-independent choice (Sect. 3.1.5)
E_MAX_EV = 13.6
GRID_MAX_EV = 16.5                         # k(E) tables reach past 13.6 eV (Paper I Fig. 9 goes to 15)
# Every 2nu and TS frequency is a multiple of 50 cm^-1, so a 50 cm^-1 grain
# counts states exactly; rho is averaged over 5 lattice points (250 cm^-1).
GRAIN_CM, WINDOW_CM = 50.0, 250.0

#: Table 3: (charge kind, "PAH" for N_H <= y or "PAHH" for N_H > y, channel) -> (E0 [eV], dS [cal/K])
CHANNELS = {("cation", "PAH", "H"): (4.8, 5.0),
            ("cation", "PAHH", "H"): (2.9, 5.0),
            ("cation", "PAHH", "H2"): (3.2, 5.0),
            ("neutral", "PAH", "H"): (4.8, 5.0),
            ("neutral", "PAHH", "H"): (1.2, 0.0),
            ("neutral", "PAHH", "H2"): (1.6, 5.0)}
DNH = {"H": -1, "H2": -2}

K_BENZENE_H = 2.2e-10          # cm^3 s^-1, C6H6+ + H (Sect. 3.4)
K_ODD_H = K_ODD_H2 = 5.0e-11   # cm^3 s^-1, odd-dehydrogenated cations (Sect. 3.4)
K_PAHH_H = 1.0e-12             # cm^3 s^-1, each step above y (Sect. 3.4)
K_REC_300K = {"benzene": 1.0e-6, "naphthalene": 3.0e-7}   # cm^3 s^-1 (LP03 Sect. 3.2.3)
K_ATTACH = 1.0e-7              # cm^3 s^-1 x N_C^3/4 (eq. 21)
SIGMA_DETACH = 2.0e-18         # cm^2 per C atom (Sect. 3.5.1)
N_PH_DETACH = 1.0e9            # photons cm^-2 s^-1 above ~3 eV (Sect. 3.5.1)
NE_OVER_NT = 1.4e-4            # (Sect. 2)

SIGMA = "lepage01_fig11"       # their Joblin-soot sigma per C, digitised (sigma_uv)
SIGMA_FILE = "pahchem/lepage01_fig11_sigma_per_c.csv"


# ---------------------------------------------------------------------------
# 2nu-RRKM (Paper I Sect. 3.1.3)
# ---------------------------------------------------------------------------

def two_nu_modes(nc, nh):
    """Reactant frequencies of C_xH_y: 3x - 6 at 1000 and 3y at 1300 cm^-1."""
    return np.array([NU_CC] * (3 * nc - 6) + [NU_CH] * (3 * nh))


def two_nu_ts(nc, nh, dS_cal=5.0):
    """Transition state: one 1300 mode removed, two turned into bends at
    250 cm^-1 (dS = 5 cal/K) or 950 cm^-1 (dS = 0)."""
    return np.array([NU_CC] * (3 * nc - 6) + [NU_CH] * (3 * nh - 3) + [TS_BEND[dS_cal]] * 2)


def path_degeneracy(nh, nh0, channel, charge=0):
    """Reaction-path degeneracy (not stated; card [dissociation], confidence D-E).

    - Aromatic H loss (N_H <= y): N_H, the equivalent H. This reproduces the
      coronene, benzene and anthracene checks.
    - sp3 H loss (N_H > y), neutral: 2 (N_H - y), the H on the CH2 groups,
      read from Fig. 9's C24H13 (thumbnail; Fig. 10 is insensitive: the
      neutral PAHH rate is saturated).
    - sp3 H loss, cation: N_H - y. Paper I Fig. 10 (digitised) fixes it: with
      2 (N_H - y) PAHH+ is +0.22-0.29 dex high from 50 C up, where the rate is
      no longer saturated; with N_H - y, within 0.15 dex at every size.
    - sp3 H2 loss: N_H - y, one per CH2 group (an assumption, E).
    """
    if nh <= nh0:
        return float(nh) if channel == "H" else 0.0
    extra = nh - nh0
    if channel == "H":
        return float(extra) if charge > 0 else 2.0 * extra
    return float(extra)


@lru_cache(maxsize=None)
def _grid(nc, nh):
    return dos.beyer_swinehart(two_nu_modes(nc, nh), GRID_MAX_EV * EV, GRAIN_CM)


@lru_cache(maxsize=None)
def rate_table(nc, nh, E0_ev, dS_cal, degeneracy):
    """(E [eV], k(E) [s^-1]) of one 2nu-RRKM channel of C_x H_y."""
    E, counts = _grid(nc, nh)
    k = rates.rrkm(E, counts, two_nu_ts(nc, nh, dS_cal), E0_ev * EV, degeneracy=degeneracy,
                   window_cm=WINDOW_CM)
    return E / EV, k


def channels(nc, nh, nh0, charge):
    """{channel: (E0, dS, sigma)} open to C_x H_{nh} at ``charge`` (Table 3 scope rule)."""
    kind = "cation" if charge > 0 else "neutral"
    scope = "PAH" if nh <= nh0 else "PAHH"
    out = {}
    for ch in ("H", "H2"):
        if (kind, scope, ch) not in CHANNELS or nh + DNH[ch] < 0:
            continue
        E0, dS = CHANNELS[(kind, scope, ch)]
        s = path_degeneracy(nh, nh0, ch, charge)
        if s > 0:
            out[ch] = (E0, dS, s)
    return out


def k_of_E(nc, nh, E0, dS, sigma, energy_ev):
    E, k = rate_table(nc, nh, E0, dS, sigma)
    return np.interp(energy_ev, E, k)


def branching(nc, nh, nh0, charge, energy_ev, stepladder=True, k_rad=K_RAD, q_ir_ev=Q_IR_EV):
    """{channel: P_i(E)}, P_i = k_i / (k_diss + k'_rad) at E = h nu (eq. 11).

    k'_rad = k_rad prod_{i=1..n} k_rad / (k_diss(E - i q_IR) + k_rad), with
    n = (E - E0)/q_IR (eq. 12). E0 is the lowest open barrier; k_diss is the
    sum over channels. ``stepladder=False`` uses k'_rad = k_rad, which Sect.
    3.1.5 calls acceptable.
    """
    E = np.atleast_1d(np.asarray(energy_ev, dtype=float))
    ch = channels(nc, nh, nh0, charge)
    if not ch:
        return {}
    k = {c: k_of_E(nc, nh, *p, E) for c, p in ch.items()}
    k_tot = sum(k.values())
    k_eff = np.full_like(E, k_rad)
    if stepladder:
        e0 = min(p[0] for p in ch.values())
        n_max = int(np.floor((E.max() - e0) / q_ir_ev))
        for i in range(1, n_max + 1):
            Ei = E - i * q_ir_ev
            kd = sum(k_of_E(nc, nh, *p, Ei) for p in ch.values())
            k_eff = np.where(Ei > e0, k_eff * k_rad / (kd + k_rad), k_eff)
    return {c: v / (k_tot + k_eff) for c, v in k.items()}


# ---------------------------------------------------------------------------
# Field, cross section, ionisation potential
# ---------------------------------------------------------------------------

def field(chi=1.0):
    """Draine (1978) at 5-13.6 eV spliced to Mathis+83 below 5 eV, times chi (Sect. 2)."""
    m, d = fields.mathis1983(), fields.draine1978()
    E = np.concatenate([m.energy_ev[m.energy_ev < 5.0], d.energy_ev])
    F = np.concatenate([m.photon_flux[m.energy_ev < 5.0], d.photon_flux]) * chi
    return fields.RadiationField(E, F, f"Le Page+01 (Draine + Mathis < 5 eV) x {chi:g}")


@lru_cache(maxsize=None)
def _fig11():
    t = pd.read_csv(_paths.get_external_data_path(SIGMA_FILE), comment="#")
    return t["energy_eV"].to_numpy(), t["sigma_per_c_Mb"].to_numpy() * 1e-18


def sigma_uv(nc, energy_ev, substitute=SIGMA):
    """UV cross section [cm^2], the same for every charge and N_H (LP03 Sect. 3.2.3).

    ``"lepage01_fig11"`` (default): their Joblin-soot fit per C atom, digitised
    from Paper I Fig. 11, x N_C; held at its 13.16 eV value up to 13.6 eV.
    Substitutes for comparison: ``"li_draine"``, the Li & Draine (2001)
    neutral sigma; ``"malloci"``, the Malloci coronene neutral sigma per C x
    N_C.
    """
    if substitute == "lepage01_fig11":
        E, s = _fig11()
        return np.interp(energy_ev, E, s, left=0.0) * nc
    if substitute == "li_draine":
        return photo.sigma_li_draine(nc, 0, energy_ev)
    if substitute == "malloci":
        return photo.sigma_malloci_coronene(0, energy_ev) * nc / 24.0
    raise ValueError(f"unknown sigma substitute {substitute!r}")


@lru_cache(maxsize=None)
def _sigma_on_grid(nc, e_min, substitute):
    # photo.sigma_li_draine re-reads its data file on every call
    return sigma_uv(nc, energy_grid(e_min), substitute)


def ionisation_potential(nc):
    """IP [eV]: not stated; Allain+96 I eq. 12, 4.8 + 10.9/a, a = 0.9 N_C^1/2 A."""
    return 4.8 + 10.9 / (0.9 * np.sqrt(nc))


def energy_grid(e_min, n=300):
    return np.linspace(e_min, E_MAX_EV, n)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def k_ion(nc, chi=1.0, substitute=SIGMA):
    """Photoionisation rate, eq. 15: int_IP^13.6 Y_ion sigma N dE [s^-1].

    The yield is eq. 14 in um^-1 (``photo.yield_lepage2001(form="lepage01")``)
    with IP_coronene = ``ionisation_potential(24)``.
    """
    ip = ionisation_potential(nc)
    E = energy_grid(ip)
    y = photo.yield_lepage2001(E, ip, ionisation_potential(24), form="lepage01")
    return field(chi).photon_rate(_sigma_on_grid(nc, ip, substitute) * y, E)


def k_photodissociation(nc, nh, nh0, charge, chi=1.0, one_minus_yion=True, stepladder=True,
                        e_min=None, substitute=SIGMA):
    """{channel: k_ph} [s^-1], eq. 11: int_e_min^13.6 P_i(E) sigma N dE.

    Two defaults depart from eq. 11 as printed (which starts at IP and has no
    yield factor). Paper I Fig. 10 fixes both:

    - ``e_min`` defaults to the bottom of the field (0.5 eV; any e_min <=
      2.8 eV gives the same within 2 %). From IP, the PAHH rates (E0 = 1.2
      eV) are 3-20x below Fig. 10.
    - ``one_minus_yion`` multiplies the neutral integrand by (1 - Y_ion). The
      caption calls neutrals less dissociative "because of the existing
      competitive photoionization channel"; without it the neutral PAH rates
      are 0.2-0.7 dex above Fig. 10.

    ``e_min="ip"`` restores the printed lower limit.
    """
    ip = ionisation_potential(nc)
    e_min = (field(chi).energy_ev.min() if e_min is None else ip if e_min == "ip" else float(e_min))
    E = energy_grid(e_min)
    P = branching(nc, nh, nh0, charge, E, stepladder=stepladder)
    w = _sigma_on_grid(nc, e_min, substitute)
    if one_minus_yion and charge == 0:
        w = w * (1.0 - photo.yield_lepage2001(E, ip, ionisation_potential(24), form="lepage01"))
    f = field(chi)
    return {c: f.photon_rate(w * p, E) for c, p in P.items()}


#: LP03 Sect. 4.3: C2H2 loss (E0 by size: 4.5 eV small PAHs, 5.5 pyrene+, 6.0 large cations;
#: dS 10 cal/K; 6 paths in compact PAHs) and C2 loss of the bare skeleton (8 eV; dS not
#: stated, 5 cal/K being the stated default "for processes for which no data are available")
C2H2_LOSS_DS, C2H2_LOSS_DEGENERACY = 10.0, 6.0
C2_LOSS = (8.0, 5.0)
H_ADDITION_FIG6, C_INSERTION_FIG6 = 2.5e-9, 3.5e-12   # s^-1 at nt = 25 (LP03 Sect. 4.3)


def carbon_loss_ts(nc, nh, dS_cal):
    """C-loss transition state (not stated): one 1000 cm^-1 mode removed and
    two more scaled to give dS(1000 K), mirroring the H-loss TS (an
    assumption, E)."""
    return rates.transition_state_for_entropy(two_nu_modes(nc, nh), NU_CC, dS_cal * CAL, n_scaled=2)


def k_carbon_loss(nc, nh, nh0, charge, E0_ev, dS_cal, degeneracy=1.0, chi=1.0, substitute=SIGMA):
    """C2 / C2H2 loss rate [s^-1], LP03 Figs. 6-7: int P_C sigma N dE over the
    whole field, P_C = k_C / (k_C + k_H + k_rad) at E = h nu, with (1 - Y_ion)
    for neutrals, as k_photodissociation."""
    E = energy_grid(field(chi).energy_ev.min())
    Eg, counts = _grid(nc, nh)
    kc = np.interp(E, Eg / EV, rates.rrkm(Eg, counts, carbon_loss_ts(nc, nh, dS_cal), E0_ev * EV,
                                           degeneracy=degeneracy, window_cm=WINDOW_CM))
    kh = sum((k_of_E(nc, nh, *p, E) for p in channels(nc, nh, nh0, charge).values()), np.zeros_like(E))
    w = _sigma_on_grid(nc, float(E[0]), substitute)
    if charge == 0:
        ip = ionisation_potential(nc)
        w = w * (1.0 - photo.yield_lepage2001(E, ip, ionisation_potential(24), form="lepage01"))
    return field(chi).photon_rate(w * kc / (kc + kh + K_RAD), E)


def k_recombination(temperature_k, n_e, law="benzene"):
    """k_rec n_e [s^-1]: size independent, (300/T)^1/2 (LP03 Sect. 3.2.3)."""
    return K_REC_300K[law] * np.sqrt(300.0 / temperature_k) * n_e


def k_attachment(nc, n_e):
    """k_e n_e [s^-1], eq. 21."""
    return K_ATTACH * nc**0.75 * n_e


def k_photodetachment(nc, chi=1.0):
    """k_p [s^-1], eq. 22 (sigma_ion N_C N_ph)."""
    return SIGMA_DETACH * nc * N_PH_DETACH * chi


def three_state_charge(k_ion_s, k_rec_s, k_e_s, k_p_s):
    """(f_cation, f_neutral, f_anion) of anion <-> neutral <-> cation at steady state."""
    plus, minus = k_ion_s / k_rec_s, k_e_s / k_p_s
    f0 = 1.0 / (1.0 + plus + minus)
    return plus * f0, f0, minus * f0


# ---------------------------------------------------------------------------
# The hydrogenation network (Paper I Sect. 4.1; LP03)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Environment:
    """A diffuse-cloud zone. ``f`` = n_H2 / n_H (Paper I, LP03)."""

    n_t: float = 100.0
    f: float = 0.5
    chi: float = 1.0
    temperature_k: float = 100.0
    ne_over_nt: float = NE_OVER_NT
    recombination: str = "naphthalene"

    @property
    def n_h(self):
        return self.n_t / (1.0 + 2.0 * self.f)

    @property
    def n_h2(self):
        return self.f * self.n_h

    @property
    def n_e(self):
        return self.ne_over_nt * self.n_t


STANDARD = Environment()      # LP03 Sect. 2 (Fig. 9 uses the naphthalene recombination; card)


class LePage01:
    """One compact C_x H_y family: Z = 0, +1, N_H = 0..2y.

    Choices where the card says "not stated":
    - H addition to cations uses the normal y in (y/x) 2.2e-10.
    - Even-dehydrogenated cations add no H2.
    - The neutral rows have no chemistry (Sect. 3.4).
    """

    def __init__(self, nc, nh0, one_minus_yion=True, stepladder=True, substitute=SIGMA):
        self.nc, self.nh0 = nc, nh0
        self.nh_values = tuple(range(0, 2 * nh0 + 1))
        self.opts = dict(one_minus_yion=one_minus_yion, stepladder=stepladder, substitute=substitute)

    def photo_rates(self, chi=1.0):
        """{(Z, N_H): {channel: k_ph}} and k_ion."""
        out = {}
        for z in (0, 1):
            for nh in self.nh_values:
                out[(z, nh)] = k_photodissociation(self.nc, nh, self.nh0, z, chi, **self.opts)
        return out, k_ion(self.nc, chi, self.opts["substitute"])

    def recombination_branching(self, nh):
        """{channel: P} of the neutral formed at E = IP (Sect. 3.3); the rest stabilises."""
        P = branching(self.nc, nh, self.nh0, 0, ionisation_potential(self.nc),
                      stepladder=self.opts["stepladder"])
        return {c: float(np.squeeze(p)) for c, p in P.items()}

    def network(self, env=STANDARD, photo_rates=None):
        pr, kion = self.photo_rates(env.chi) if photo_rates is None else photo_rates
        nh = np.array(self.nh_values)
        y, x = self.nh0, self.nc
        net = PAHNetwork(charges=(0, 1), nh_values=self.nh_values)
        for ch, dnh in DNH.items():
            net.add(f"photodissociation {ch}",
                    np.array([[pr[(z, n)].get(ch, 0.0) for n in self.nh_values] for z in (0, 1)]),
                    dnh=dnh, h2_released=1 if ch == "H2" else 0)
        net.add("photoionisation", np.array([np.full(nh.size, kion), np.zeros(nh.size)]), dz=+1)
        krec = k_recombination(env.temperature_k, env.n_e, env.recombination)
        P = [self.recombination_branching(n) for n in self.nh_values]
        stable = np.array([1.0 - sum(p.values()) for p in P])
        net.add("recombination", np.array([np.zeros(nh.size), krec * stable]), dz=-1)
        for ch, dnh in DNH.items():
            net.add(f"dissociative recombination {ch}",
                    np.array([np.zeros(nh.size), krec * np.array([p.get(ch, 0.0) for p in P])]),
                    dz=-1, dnh=dnh, h2_released=1 if ch == "H2" else 0)
        deficit = y - nh
        add_h = np.select([nh == y, (deficit > 0) & (deficit % 2 == 0), (deficit > 0) & (deficit % 2 == 1),
                           (nh > y) & (nh < 2 * y)],
                          [(y / x) * K_BENZENE_H, (y / x) * K_BENZENE_H, K_ODD_H, K_PAHH_H], 0.0)
        net.add("H addition, cations", np.array([np.zeros(nh.size), add_h * env.n_h]), dnh=+1)
        add_h2 = np.where((deficit > 0) & (deficit % 2 == 1), K_ODD_H2, 0.0)
        net.add("H2 addition, odd cations", np.array([np.zeros(nh.size), add_h2 * env.n_h2]), dnh=+2)
        return net

    def steady_state(self, env=STANDARD, photo_rates=None):
        return self.network(env, photo_rates).steady_state()
