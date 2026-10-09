"""Energy deposited in a PAH by a fast charged particle (cosmic-ray nuclei and electrons).

A particle of charge z, speed beta c and kinetic energy T crossing a PAH
excites its electrons. The model is the Bethe theory written over the
molecule's own optical oscillator strength df/dW. It is the delta-oscillator
model of Liljequist (1983) and Ashley (1988): each oscillator of energy W
takes part in two kinds of collision.

- **Distant (dipole) collisions** with momentum transfer below the
  oscillator's. The field of the passing charge acts as a pulse of virtual
  photons (Fermi; Weizsaecker-Williams), and the molecule absorbs one of energy W:

      dsigma_d/dW = (2 pi z^2 e^4 / m c^2 beta^2) (df/dW) / W * L_d(W),
      L_d = ln(2 m c^2 beta^2 gamma^2 / W) - beta^2          (heavy particles)
      L_d = ln(W / Q_-),  Q_- = (sqrt T - sqrt(T - W))^2       (electrons, T < 10 keV)

  The outcome is the one an absorbed photon of energy W would have, so the
  photoionisation yields of ``photo`` apply unchanged.
- **Close (binary) collisions** in which the oscillator's electron recoils as
  if free, with energy Q from W to Q_max. The spin factor is
  (1 - beta^2 Q / Q_max) for nuclei. For electrons, the Moeller exchange form
  is used with Q_max = T/2:

      dsigma_c/dQ = (2 pi z^2 e^4 / m c^2 beta^2) (df/dW) g(Q) / Q^2.

  The electron leaves (a delta ray). The molecule keeps a hole of energy
  max(W, IP), i.e. E* = max(W, IP) - IP above the ion's ground state.

Summed over oscillators, the two parts give the Bethe stopping number
ln(2 m c^2 beta^2 gamma^2 / I) - beta^2, with ln I = int f ln W / int f.
Bethe needs no other input. Its validity at low speed is set by the measured
stopping of polystyrene, an aromatic CH polymer (``low_speed_correction``).

df/dW is the Li & Draine (2001) cross section of the PAH below ``W_JOIN_EV``
and the Henke et al. (1993) atomic C and H photoabsorption above it. It obeys
the Thomas-Reiche-Kuhn sum rule int f dW = 6 N_C + N_H to 1-4% for
C14-C96 (``tests/test_pahchem_impact.py``).

Energies are in eV and cross sections in cm^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pycalima import _paths
from pycalima.pahchem import photo

E2_EV_CM = 1.439964548e-7                         # e^2 [eV cm]
TWO_PI_E4 = 2.0 * np.pi * E2_EV_CM**2             # [eV^2 cm^2]
ME_EV = 510998.95                                 # m_e c^2
AMU_EV = 931494102.4
R_E_CM = 2.8179403262e-13
HC_EV_CM = 1.239841984e-4
#: f per (cm^2 eV): df/dW = F_PER_CM2_EV * sigma_abs(W)  (m_e c / pi e^2 h)
F_PER_CM2_EV = 1.0 / (np.pi * R_E_CM * HC_EV_CM)
#: Li & Draine (2001) below this energy, Henke atomic C+H above (the two agree
#: to 13% here for C24-C96; 25% at 30 and 50 eV)
W_JOIN_EV = 40.0
#: polystyrene (C8H8)n, the STAR reference: I, electrons and mass per C8H8
POLYSTYRENE = {"I_ev": 68.7, "electrons": 56, "mass_g": 104.15 / 6.02214076e23}


@dataclass(frozen=True)
class Projectile:
    """A charged projectile: name, charge number z, rest energy [eV]."""

    name: str
    z: int
    rest_ev: float

    @property
    def is_electron(self) -> bool:
        return self.rest_ev == ME_EV

    @property
    def nucleons(self) -> float:
        return self.rest_ev / AMU_EV


ELECTRON = Projectile("e", 1, ME_EV)
PROTON = Projectile("H", 1, 938272088.2)
ALPHA = Projectile("He", 2, 3727379411.0)


def nucleus(name: str, z: int, mass_amu: float) -> Projectile:
    """A bare nucleus of charge z and mass ``mass_amu``."""
    return Projectile(name, z, mass_amu * AMU_EV)


# --- oscillator strength ----------------------------------------------------

@lru_cache(maxsize=None)
def _henke(element: str):
    d = np.loadtxt(_paths.get_external_data_path(f"pahchem/henke1993_{element}_f2.csv"),
                   delimiter=",", comments="#")
    E = d[:, 0]
    return E, 2.0 * R_E_CM * (HC_EV_CM / E) * d[:, 1]


def henke_sigma(element: str, energy_ev) -> np.ndarray:
    """Atomic photoabsorption cross section [cm^2] of ``carbon`` or ``hydrogen``
    (Henke+93, 10 eV - 30 keV; log-log interpolation, zero outside)."""
    E, s = _henke(element)
    w = np.asarray(energy_ev, dtype=float)
    out = np.exp(np.interp(np.log(w), np.log(E), np.log(s)))
    return np.where((w >= E[0]) & (w <= E[-1]), out, 0.0)


def oscillator_strength(nc: int, nh: int, charge: int, energy_ev) -> np.ndarray:
    """df/dW [eV^-1] of C_{nc}H_{nh} at charge Z: Li & Draine below W_JOIN_EV,
    N_C sigma_C + N_H sigma_H (Henke) above."""
    w = np.asarray(energy_ev, dtype=float)
    low = w < W_JOIN_EV
    sigma = np.empty_like(w)
    if low.any():
        sigma[low] = photo.sigma_li_draine(nc, charge, w[low])
    sigma[~low] = nc * henke_sigma("carbon", w[~low]) + nh * henke_sigma("hydrogen", w[~low])
    return F_PER_CM2_EV * sigma


def w_grid(n: int = 1200, w_min: float = 0.5, w_max: float = 3.0e4) -> np.ndarray:
    """The default log grid of oscillator energies [eV]."""
    return np.geomspace(w_min, w_max, n)


def mean_excitation_energy(nc: int, nh: int, charge: int = 0, w=None) -> float:
    """Bethe's I [eV]: ln I = int f ln W dW / int f dW."""
    w = w_grid() if w is None else w
    f = oscillator_strength(nc, nh, charge, w)
    return float(np.exp(np.trapezoid(f * np.log(w), w) / np.trapezoid(f, w)))


# --- kinematics and the two collision types ---------------------------------

def kinematics(proj: Projectile, t_ev):
    """(beta^2, gamma, Q_max) of a projectile of kinetic energy T [eV]."""
    t = np.asarray(t_ev, dtype=float)
    gamma = 1.0 + t / proj.rest_ev
    beta2 = 1.0 - 1.0 / gamma**2
    if proj.is_electron:
        q_max = t / 2.0
    else:
        r = ME_EV / proj.rest_ev
        q_max = 2.0 * ME_EV * beta2 * gamma**2 / (1.0 + 2.0 * gamma * r + r * r)
    return beta2, gamma, q_max


def _distant_log(proj, t, beta2, gamma, w):
    if proj.is_electron and t < 1.0e4:
        q_minus = (np.sqrt(t) - np.sqrt(np.clip(t - w, 0.0, None))) ** 2
        with np.errstate(divide="ignore"):
            L = np.where(w < t, np.log(w / q_minus), 0.0)
    else:
        L = np.log(2.0 * ME_EV * beta2 * gamma**2 / w) - beta2
    return np.clip(L, 0.0, None)


def _close_number_and_loss(proj, t, beta2, q_max, w):
    """(int g/Q^2 dQ, int g/Q dQ) over W < Q < Q_max per oscillator W."""
    ok = w < q_max
    wq = np.where(ok, w, 0.5 * q_max)
    if proj.is_electron:
        n = 1.0 / wq - 1.0 / (t - wq) + np.log(wq / (t - wq)) / t
        e = np.log(t / (2.0 * wq)) + 2.0 - t / (t - wq) + 2.0 * np.log(t / (2.0 * (t - wq)))
    else:
        n = 1.0 / wq - 1.0 / q_max - beta2 / q_max * np.log(q_max / wq)
        e = np.log(q_max / wq) - beta2 * (1.0 - wq / q_max)
    return np.where(ok, np.clip(n, 0.0, None), 0.0), np.where(ok, np.clip(e, 0.0, None), 0.0)


def effective_charge(proj: Projectile, beta2) -> np.ndarray:
    """Barkas effective charge z* = z [1 - exp(-125 beta z^(-2/3))] (Micelotta+11 eq. 9)."""
    return proj.z * (1.0 - np.exp(-125.0 * np.sqrt(beta2) * proj.z ** (-2.0 / 3.0)))


@dataclass(frozen=True)
class Collisions:
    """Collision cross sections of one projectile at one energy, per oscillator energy W.

    ``distant`` and ``close`` are dsigma/dW [cm^2 eV^-1] (the close part
    integrated over the recoil energy Q). ``close_loss`` is the projectile's
    mean energy loss in a close collision on oscillator W [eV].
    """

    w: np.ndarray
    distant: np.ndarray
    close: np.ndarray
    close_loss: np.ndarray

    @property
    def total(self) -> float:
        """Collisions per unit flux [cm^2]."""
        return float(np.trapezoid(self.distant + self.close, self.w))

    @property
    def stopping(self) -> float:
        """Energy lost per unit flux [eV cm^2]."""
        return float(np.trapezoid(self.distant * self.w + self.close * self.close_loss, self.w))


#: electrons below this energy [eV] get Kim's binary-encounter scaling
BE_SCALING_MAX_EV = 1.0e4


def collisions(proj: Projectile, t_ev: float, df_dw, w, correction: float = 1.0,
               ip_ev: float | None = None) -> Collisions:
    """Distant and close collisions of ``proj`` at kinetic energy T on oscillators df/dW(w).

    ``correction`` multiplies both. ``impact_collisions`` uses it for the
    low-speed correction and the heavy-ion effective charge. For electrons
    below BE_SCALING_MAX_EV, with ``ip_ev`` given, both are multiplied by
    Kim's binary-encounter factor T / (T + IP + W) (Kim 2001, PRA 64, 032713).
    The plane-wave Born form overestimates slow-electron cross sections, and
    this factor stands in for the acceleration of the electron by the target
    field. With it, the stopping of 80 eV - 2 keV electrons is within 15% of
    Joy (1995) carbon; it is 1.35x at 50 eV and 2.3x at 30 eV.
    """
    t = float(t_ev)
    beta2, gamma, q_max = (float(x) for x in kinematics(proj, t))
    pre = correction * TWO_PI_E4 * proj.z**2 / (ME_EV * beta2) * np.asarray(df_dw)
    if proj.is_electron and ip_ev is not None and t < BE_SCALING_MAX_EV:
        pre = pre * t / (t + ip_ev + w)
    distant = pre / w * _distant_log(proj, t, beta2, gamma, w)
    n, e = _close_number_and_loss(proj, t, beta2, q_max, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(n > 0, e / n, 0.0)
    return Collisions(w, distant, pre * n, loss)


# --- stopping: the Bethe limit and the measured polystyrene -----------------

def bethe_stopping_per_electron(proj: Projectile, t_ev, i_ev: float) -> np.ndarray:
    """Bethe stopping cross section per target electron [eV cm^2], no corrections.

    Nuclei: (4 pi z^2 e^4 / m c^2 beta^2) [ln(2 m c^2 beta^2 gamma^2 / I) - beta^2], in the
    form the delta-oscillator model reduces to. Electrons: the Rohrlich-Carlson
    form, (2 pi e^4 / m c^2 beta^2) [ln(tau^2 (tau + 2) / (2 (I/mc^2)^2)) + F^-(tau)].
    """
    t = np.asarray(t_ev, dtype=float)
    beta2, gamma, q_max = kinematics(proj, t)
    if proj.is_electron:
        tau = t / ME_EV
        f_minus = 1.0 - beta2 + (tau**2 / 8.0 - (2.0 * tau + 1.0) * np.log(2.0)) / (tau + 1.0) ** 2
        L = np.log(tau**2 * (tau + 2.0) / (2.0 * (i_ev / ME_EV) ** 2)) + f_minus
        return TWO_PI_E4 / (ME_EV * beta2) * L
    L = 0.5 * np.log(2.0 * ME_EV * beta2 * gamma**2 * q_max / i_ev**2) - beta2
    return 2.0 * TWO_PI_E4 * proj.z**2 / (ME_EV * beta2) * L


@lru_cache(maxsize=None)
def _star(prog: str):
    d = np.loadtxt(_paths.get_external_data_path(f"pahchem/nist_{prog}_polystyrene.csv"),
                   delimiter=",", comments="#")
    T = d[:, 0] * 1e6                                 # [eV]
    S = d[:, 1] * 1e6 * POLYSTYRENE["mass_g"] / POLYSTYRENE["electrons"]   # per electron [eV cm^2]
    if prog == "estar":
        # The ESTAR density effect delta is a property of the condensed medium,
        # not of an isolated molecule; put it back: Delta S = (2 pi e^4 / m c^2 beta^2) delta.
        beta2 = kinematics(ELECTRON, T)[0]
        S = S + TWO_PI_E4 / (ME_EV * beta2) * d[:, 6]
    return T, S


def polystyrene_stopping_per_electron(prog: str, t_ev) -> np.ndarray:
    """NIST STAR electronic (collision) stopping of polystyrene per electron [eV cm^2].

    ``prog``: ``pstar`` (protons), ``astar`` (alphas), ``estar`` (electrons,
    with the density effect removed). NaN outside the tables.
    """
    T, S = _star(prog)
    return np.exp(np.interp(np.log(np.asarray(t_ev, dtype=float)), np.log(T), np.log(S), left=np.nan, right=np.nan))


#: below this speed (kinetic energy per nucleon, eV) the Bethe log of polystyrene
#: is under 1 and the low-speed correction is not used: impact rates are zero
T_MIN_PER_U_EV = 1.0e5


def low_speed_correction(proj: Projectile, t_ev: float) -> float:
    """S_STAR / S_Bethe for polystyrene at the projectile's speed.

    It carries the shell, Barkas and Bloch corrections and the effective
    charge that Bethe omits, as measured, from aromatic CH to the PAH
    (an assumption). Protons and nuclei use PSTAR at the same speed, times
    (z*/z)^2 for nuclei heavier than He. Alphas use ASTAR. Electrons use ESTAR
    from 10 keV; below that the correction is 1. It is 1 above the tables
    (10 GeV protons, 4 GeV alphas, 1 GeV electrons) and 0 below
    T_MIN_PER_U_EV.
    """
    t = float(t_ev)
    if proj.is_electron:
        T, _ = _star("estar")
        if t < T[0] or t > T[-1]:
            return 1.0
        return float(polystyrene_stopping_per_electron("estar", t)
                     / bethe_stopping_per_electron(ELECTRON, t, POLYSTYRENE["I_ev"]))
    t_u = t / proj.nucleons
    if t_u < T_MIN_PER_U_EV:
        return 0.0
    if proj.z == 2:
        prog, ref, tt = "astar", ALPHA, t_u * ALPHA.nucleons
    else:
        prog, ref, tt = "pstar", PROTON, t_u * PROTON.nucleons
    T, _ = _star(prog)
    if tt > T[-1]:
        c = 1.0
    else:
        c = float(polystyrene_stopping_per_electron(prog, tt) / bethe_stopping_per_electron(ref, tt, POLYSTYRENE["I_ev"]))
    if proj.z > 2:
        beta2 = float(kinematics(proj, t)[0])
        c *= float(effective_charge(proj, beta2) / proj.z) ** 2
    return c


def impact_collisions(proj: Projectile, t_ev: float, nc: int, nh: int, charge: int = 0,
                      ip_ev: float | None = None, w=None) -> Collisions:
    """``collisions`` on C_{nc}H_{nh}, with the low-speed correction and (for slow
    electrons, given ``ip_ev``) the binary-encounter scaling applied."""
    w = w_grid() if w is None else w
    return collisions(proj, t_ev, oscillator_strength(nc, nh, charge, w), w,
                      correction=low_speed_correction(proj, t_ev), ip_ev=ip_ev)


# --- what a collision leaves behind -----------------------------------------

#: carbon K edge [eV] (Henke+93)
K_EDGE_EV = 284.2
#: Energy of the final state after a carbon K hole relaxes by KVV Auger decay,
#: above the neutral ground state: C 1s binding ~290 eV in aromatics minus the
#: KVV Auger peak of graphitic carbon, ~263 eV. The two valence holes keep
#: K_HOLE_FINAL_EV - IP1 - IP2 as internal energy. This is an assumption with
#: confidence D; K-shell collisions are <1% of all collisions.
K_HOLE_FINAL_EV = 27.0
PARTITIONS = ("uniform", "none", "all")


@dataclass(frozen=True)
class Deposition:
    """Rate densities of collision outcomes against the internal energy E* left in the product.

    ``rates[dz]`` [input units per eV] on the bin centres ``e`` (width
    ``de``), for a charge change dz = 0, +1, +2. ``overflow[dz]`` is the part
    with E* beyond the grid. Per unit flux the units are cm^2 eV^-1; after
    ``impact_rates``, s^-1 eV^-1 per molecule.
    """

    e: np.ndarray
    de: float
    rates: dict
    overflow: dict

    def total(self, dz=None) -> float:
        keys = self.rates if dz is None else [dz]
        return float(sum(self.rates[k].sum() * self.de + self.overflow[k] for k in keys))

    def power(self) -> float:
        """Internal energy deposited per unit time (in the grid) [eV x input units]."""
        return float(sum((self.rates[k] * self.e).sum() * self.de for k in self.rates))

    def __add__(self, other: "Deposition") -> "Deposition":
        return Deposition(self.e, self.de, {k: self.rates[k] + other.rates[k] for k in self.rates},
                          {k: self.overflow[k] + other.overflow[k] for k in self.overflow})

    def scaled(self, s: float) -> "Deposition":
        return Deposition(self.e, self.de, {k: s * v for k, v in self.rates.items()},
                          {k: s * v for k, v in self.overflow.items()})


def _trapezoid_weights(x):
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def deposition(coll: Collisions, ip_ev: float, ip2_ev: float, yield_, e_max: float = 200.0,
               de: float = 0.25, partition: str = "uniform") -> Deposition:
    """Map collisions on oscillators W to (charge change, internal energy E*).

    - **Distant collisions** have the outcome of a photon of energy W.
      - Below IP: E* = W.
      - Above IP: ionisation with the photoionisation yield ``yield_(W)``.
        The ion keeps E* of W - IP minus the electron's energy:
        - ``partition="uniform"``: E* uniform on [0, W - IP];
        - ``"none"``: E* = 0, Montillaud+13's convention for photons;
        - ``"all"``: E* = W - IP.

        The three bracket the unknown photoelectron spectrum.
      - Otherwise E* = W.
    - **Close collisions** ionise. The hole is kept: E* = max(W, IP) - IP.
    - **W above the carbon K edge**, either kind: Auger decay gives dz = +2 with
      E* = K_HOLE_FINAL_EV - IP - IP2.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, not {partition!r}")
    w = coll.w
    tw = _trapezoid_weights(w)
    edges = np.arange(0.0, e_max + 0.5 * de, de)
    centres = 0.5 * (edges[1:] + edges[:-1])
    k_shell = w >= K_EDGE_EV
    y = np.where((w >= ip_ev) & ~k_shell, np.clip(yield_(w), 0.0, 1.0), 0.0)
    d, c = coll.distant * tw, coll.close * tw

    def hist(x, q):
        h, _ = np.histogram(x, bins=edges, weights=q)
        return h / de, float(q[x >= edges[-1]].sum())

    r0, o0 = hist(w, np.where(k_shell, 0.0, d * (1.0 - y)))
    r1, o1 = hist(np.maximum(w, ip_ev) - ip_ev, np.where(k_shell, 0.0, c))
    x = np.clip(w - ip_ev, 0.0, None)
    q = d * y
    if partition == "uniform":
        span = np.where(x > 0, x, 1.0)
        overlap = np.clip(np.minimum(edges[1:][None, :], x[:, None]) - edges[:-1][None, :], 0.0, de)
        r1 = r1 + (q / span) @ overlap / de
        o1 += float((q * np.clip(x - edges[-1], 0.0, None) / span).sum())
    else:
        rr, oo = hist(x if partition == "all" else np.zeros_like(x), q)
        r1, o1 = r1 + rr, o1 + oo
    e_k = max(K_HOLE_FINAL_EV - ip_ev - ip2_ev, 0.0)
    r2, o2 = hist(np.full_like(w, e_k), np.where(k_shell, d + c, 0.0))
    return Deposition(centres, de, {0: r0, 1: r1, 2: r2}, {0: o0, 1: o1, 2: o2})


def impact_rates(nc: int, nh: int, charge: int, ip_ev: float, ip2_ev: float, spectra, yield_,
                 t_grid=None, w=None, **kwargs) -> dict:
    """{projectile name: Deposition [s^-1 eV^-1 per molecule]} for ``spectra``.

    ``spectra``: [(Projectile, j(T))] with j in particles eV^-1 s^-1 cm^-2 sr^-1
    (``cosmic_rays.projectile_spectra``), integrated as 4 pi int j D dT.
    Single collisions per passage are assumed; ``passage_multiplicity`` says
    where that fails. ``kwargs`` go to ``deposition``.
    """
    w = w_grid() if w is None else w
    df = oscillator_strength(nc, nh, charge, w)
    out = {}
    for proj, j in spectra:
        t = (np.geomspace(max(T_MIN_PER_U_EV * proj.nucleons, 1.0e5), 1.0e12 * proj.nucleons, 120)
             if not proj.is_electron else np.geomspace(1.0e3, 1.0e12, 160)) if t_grid is None else t_grid
        jt = np.asarray(j(t), dtype=float)
        tw = _trapezoid_weights(np.log(t)) * t * jt * 4.0 * np.pi
        acc = None
        for ti, wt in zip(t, tw):
            if wt <= 0.0:
                continue
            c = collisions(proj, ti, df, w, correction=low_speed_correction(proj, ti), ip_ev=ip_ev)
            dep = deposition(c, ip_ev, ip2_ev, yield_, **kwargs).scaled(wt)
            acc = dep if acc is None else acc + dep
        out[proj.name] = acc
    return out


def disc_radius_cm(nc: int) -> float:
    """PAH disc radius 0.9 sqrt(N_C) A (Omont 1986), as in Micelotta+11."""
    return 0.9e-8 * np.sqrt(nc)


#: thickness of the PAH electron cloud [cm] (Micelotta+10b, +11)
DISC_THICKNESS_CM = 4.31e-8


def passage_multiplicity(coll: Collisions, nc: int) -> float:
    """Mean number of collisions per passage, averaged over isotropic passages.

    That is sigma_tot / <sigma_g> with <sigma_g> = S/4 for the disc (Cauchy).
    Above ~0.3, several collisions in one passage add their energy, and the
    single-collision deposition underestimates the high-E* tail.
    """
    r = disc_radius_cm(nc)
    area = 2.0 * np.pi * r * r + 2.0 * np.pi * r * DISC_THICKNESS_CM
    return coll.total / (area / 4.0)
