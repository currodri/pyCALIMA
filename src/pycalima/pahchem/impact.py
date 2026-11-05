"""Energy deposited in a PAH by a fast charged particle (cosmic-ray nuclei and electrons).

A particle of charge z, speed beta c and kinetic energy T crossing a PAH
excites its electrons. The model is the Bethe theory written over the
molecule's own optical oscillator strength df/dW. It is a delta-oscillator
model in the manner of Liljequist (1983): each oscillator of energy W takes
part in two kinds of collision.

- **Distant (dipole) collisions** with momentum transfer below the
  oscillator's. The field of the passing charge acts as a pulse of virtual
  photons (Fermi; Weizsaecker-Williams), and the molecule absorbs one of energy W::

      dsigma_d/dW = (2 pi z^2 e^4 / m c^2 beta^2) (df/dW) / W * L_d(W),
      L_d = ln(2 m c^2 beta^2 gamma^2 / W) - beta^2          (heavy particles)
      L_d = ln(W / Q_-),  Q_- = (sqrt T - sqrt(T - W))^2       (electrons, T < 10 keV)

  The outcome is the one an absorbed photon of energy W would have, so the
  photoionisation yields of ``photo`` apply unchanged.
- **Close (binary) collisions** in which the oscillator's electron recoils as
  if free, with energy Q from W to Q_max. The spin factor is
  (1 - beta^2 Q / Q_max) for nuclei. For electrons, the Moeller exchange form
  is used with Q_max = T/2::

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

from scipy.special import gammainc, gammaln

from pycalima import _paths
from pycalima.pahchem import photo

E2_EV_CM = 1.439964548e-7                         # e^2 [eV cm]
C_CM_S = 2.99792458e10                            # speed of light [cm s^-1]
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
    #: factor on the ionising part of each oscillator's collisions (``deposition``): Kim's ionisation scaling
    #: over the excitation scaling that ``distant`` and ``close`` carry, for slow electrons; 1 otherwise
    ion_scale: np.ndarray | None = None
    #: (Projectile, T [eV]) of ``collisions``, so ``deposition`` can tell the close collisions below the IP that
    #: cannot ionise; None for hand-made collisions (every close collision then ionises)
    kinematics: tuple | None = None

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
#: orbital kinetic energy over binding energy, U/B, in Kim's ionisation scaling T / (T + B + U): the median of the
#: RHF valence orbitals of naphthalene-circumcoronene (diagnostics/pahchem/beb_hf_pah.py), and 1.43 for C 1s
U_OVER_B_VALENCE, U_OVER_B_K_SHELL = 2.0, 1.43


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

    That is Kim's scaling for excitation. For ionisation Kim's form (BEB) is
    T / (T + B + U), with B = max(W, IP) the binding and U = U/B x B the
    orbital kinetic energy (U/B ~ 2 for PAH valence orbitals). ``ion_scale``
    holds the ratio of the two; ``deposition`` applies it to the ionising
    collisions only (distant ones times the yield, every close one), so the
    excitations and ``stopping`` keep T / (T + IP + W). With it, sigma_ion /
    BEB(HF) over 15 eV - 5 keV is 0.99-1.41 for anthracene, 1.10-1.55 for
    coronene and 1.39-1.80 for C54H18, against 1.36-2.30, 1.28-2.21 and
    1.40-2.58 before (pahchem-audit, C5; C4 also lowers anthracene's). The
    ~1.2-1.4x left at keV is the Li & Draine dipole strength, not this factor.
    """
    t = float(t_ev)
    beta2, gamma, q_max = (float(x) for x in kinematics(proj, t))
    pre = correction * TWO_PI_E4 * proj.z**2 / (ME_EV * beta2) * np.asarray(df_dw)
    ion_scale = None
    if proj.is_electron and ip_ev is not None and t < BE_SCALING_MAX_EV:
        pre = pre * t / (t + ip_ev + w)
        b = np.maximum(w, ip_ev)
        u = np.where(w >= K_EDGE_EV, U_OVER_B_K_SHELL, U_OVER_B_VALENCE) * b
        ion_scale = (t + ip_ev + w) / (t + b + u)
    distant = pre / w * _distant_log(proj, t, beta2, gamma, w)
    n, e = _close_number_and_loss(proj, t, beta2, q_max, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(n > 0, e / n, 0.0)
    return Collisions(w, distant, pre * n, loss, ion_scale, (proj, t))


def close_ionising(coll: Collisions, ip_ev: float):
    """(share, q_exc) per oscillator W: the share of its close collisions whose recoil Q exceeds ``ip_ev`` and so frees
    the electron, n(IP) / n(W) on W < IP (1 above; 0 where Q_max = T/2 or the nuclei's Q_max is below IP), and the mean
    recoil of the others, int_W^IP Q dsigma / int_W^IP dsigma, which excites the molecule. Hand-made collisions without
    ``kinematics``: share 1. Until 2026-11-05 every close collision ionised (pahchem audit C1: 22 % of the ionisations
    of C54H18 2+ by 34 eV electrons were false, 94 % at +10)."""
    w = coll.w
    if coll.kinematics is None:
        return np.ones_like(w), w
    proj, t = coll.kinematics
    beta2, _, q_max = (float(x) for x in kinematics(proj, t))
    n_w, e_w = _close_number_and_loss(proj, t, beta2, q_max, w)
    n_ip, e_ip = (float(x[0]) for x in _close_number_and_loss(proj, t, beta2, q_max, np.array([float(ip_ev)])))
    below = (w < ip_ev) & (n_w > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(below, np.minimum(n_ip, n_w) / np.where(n_w > 0.0, n_w, 1.0), 1.0)
        dn = n_w - n_ip
        q_exc = np.where(below & (dn > 0.0), (e_w - np.minimum(e_ip, e_w)) / np.where(dn > 0.0, dn, 1.0), w)
    return share, q_exc


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
#: Mean excitation of the doubly charged ion after a carbon K hole relaxes by KVV Auger decay, above its ground state:
#: "an average excitation of about 17.3 eV" over the benzene Auger spectrum (Huo+23, EPJD 77, 181, Sect. 3, citing
#: Carniato+20); the spectrum's peak alone gives ~14 eV (1s IE 290.42 eV, peak ~252 eV, dication ground state 24.65 eV;
#: Jayadev+23); Lange+25 quote ~20 eV. No PAH Auger spectrum exists. One value for the collisions here and for the
#: photoabsorptions of ``population`` (2026-11-03; before, 27 eV above the neutral less IP1 + IP2, clipped at 0).
K_AUGER_EXCITATION_EV = 17.3
PARTITIONS = ("measured", "uniform", "none", "all")


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
               de: float = 0.25, partition: str = "measured", reference_ev=None, hole=None) -> Deposition:
    """Map collisions on oscillators W to (charge change, internal energy E*).

    - **Distant collisions** have the outcome of a photon of energy W.

      - Below IP: E* = W.
      - Above IP: ionisation with the photoionisation yield ``yield_(W)``.
        The ion keeps E* of W - IP minus the electron's energy:

        - ``partition="measured"`` (the default since 2026-11-04): the mean hole of a photoionisation at W,
          ``photo.photoelectron_energy`` (``reference_ev`` as there), as the population's photons leave it
          (``Params.ionisation_heating``);
        - ``"uniform"``: E* uniform on [0, W - IP] (the default before: ~(W - IP) / 2, against a ~6 eV hole);
        - ``"none"``: E* = 0, Montillaud+13's convention for photons;
        - ``"all"``: E* = W - IP.

        The last three bracket the photoelectron spectrum, measured only on neutrals to 22 eV. ``hole``, a callable of W,
        replaces the partition: an anion's photodetachment hole (``population.impact_ladder``; pahchem audit m7, an anion
        took the neutral's ~(1 - 0.46...0.58)(W - EA) until 2026-11-05).
      - Otherwise E* = W.
    - **Close collisions** whose recoil Q exceeds IP ionise, and the hole is kept: E* = max(W, IP) - IP. On an oscillator
      W < IP only the share n(IP) / n(W) does (``close_ionising``); the rest excite the molecule by their mean Q (dz = 0,
      without Kim's ionisation scaling). Every close collision ionised until 2026-11-05 (pahchem audit C1).
    - **W above the carbon K edge**, either kind: Auger decay gives dz = +2 with
      E* = K_AUGER_EXCITATION_EV. (``ip2_ev`` is no longer used: it set the K hole's energy before 2026-11-03.)

    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, not {partition!r}")
    w = coll.w
    tw = _trapezoid_weights(w)
    edges = np.arange(0.0, e_max + 0.5 * de, de)
    centres = 0.5 * (edges[1:] + edges[:-1])
    k_shell = w >= K_EDGE_EV
    y_all = np.asarray(yield_(w), dtype=float)
    if not np.all(np.isfinite(y_all)) or np.any(y_all < -1e-12) or np.any(y_all > 1.0 + 1e-12):
        raise ValueError("photoionisation yield outside [0, 1] beyond round-off")
    y = np.where((w >= ip_ev) & ~k_shell, np.clip(y_all, 0.0, 1.0), 0.0)        # the round-off only
    s = np.ones_like(w) if coll.ion_scale is None else coll.ion_scale           # Kim ionisation scaling (C5)
    d, c, di = coll.distant * tw, coll.close * tw * s, coll.distant * tw * s
    share, q_exc = close_ionising(coll, ip_ev)

    def hist(x, q):
        h, _ = np.histogram(x, bins=edges, weights=q)
        return h / de, float(q[x >= edges[-1]].sum())

    r0, o0 = hist(w, np.where(k_shell, 0.0, d * (1.0 - y)))
    rx, ox = hist(q_exc, np.where(k_shell, 0.0, coll.close * tw * (1.0 - share)))   # close excitations below the IP
    r0, o0 = r0 + rx, o0 + ox
    r1, o1 = hist(np.maximum(w, ip_ev) - ip_ev, np.where(k_shell, 0.0, c * share))
    x = np.clip(w - ip_ev, 0.0, None)
    q = di * y
    if partition == "uniform" and hole is None:
        span = np.where(x > 0, x, 1.0)
        overlap = np.clip(np.minimum(edges[1:][None, :], x[:, None]) - edges[:-1][None, :], 0.0, de)
        r1 = r1 + (q / span) @ overlap / de
        o1 += float((q * np.clip(x - edges[-1], 0.0, None) / span).sum())
    else:
        if hole is not None:
            xe = np.asarray(hole(w), dtype=float)
        elif partition == "measured":
            xe = photo.photoelectron_energy(w, ip_ev, reference_ev=reference_ev)[1]
        else:
            xe = x if partition == "all" else np.zeros_like(x)
        rr, oo = hist(xe, q)
        r1, o1 = r1 + rr, o1 + oo
    r2, o2 = hist(np.full_like(w, K_AUGER_EXCITATION_EV), np.where(k_shell, di + c, 0.0))
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




@dataclass(frozen=True)
class Passages:
    """Outcomes of passages through the molecule, per unit intensity j [per sr].

    ``P[dz, k]`` [cm^2 sr eV^-1] is the rate density of passages that end at
    total charge change dz with total internal energy ``e[k]``. ``beyond`` is
    the passage rate with E* >= e_max or dz > dz_max [cm^2 sr]: destruction
    by any reading. ``passages`` [cm^2 sr] counts every crossing, including
    those without a collision. ``collisions`` [cm^2 sr] is 4 pi sigma_tot.
    Multiply by j(T) and integrate over T for rates.
    """

    e: np.ndarray
    de: float
    P: np.ndarray
    beyond: float
    passages: float
    collisions: float


def passage_outcomes(coll: Collisions, nc: int, ip_ev: float, ip2_ev: float, yield_, de: float = 1.0,
                     e_max: float = 512.0, dz_max: int = 15, n_theta: int = 24, tol: float = 1e-12, chain=None,
                     **kwargs) -> Passages:
    """Several collisions in one passage, summed.

    The geometry is Micelotta+11's disc. A passage at angle theta to the normal
    crosses the projected area sigma_g = pi R^2 cos theta + 2 R d sin theta.
    It makes Poisson(nu) collisions, nu = sigma_tot / sigma_g (a uniform
    electron column over the projection). Passages from an isotropic intensity
    are weighted by sigma_g sin theta. One collision's (dz, E*) follows
    ``deposition``, on the grid points of ``_edge_split`` (E* = k de, exact in
    the mean over the n-fold sums; bin centres before 2026-11-03, de/2 low per
    extra collision).

    ``chain`` = [(Collisions, IP, IP2, yield) with 0, 1, ... charges added]: a collision after k ionisations takes the
    k-th entry (the last for every k beyond), so the k-th ionisation costs IP(z + k) (``_poisson_chain``); None: every
    collision is ``coll``'s, at ``ip_ev``. The sums are truncated to the grid, and the mass that leaves it goes to
    ``beyond``. The mixture over theta is then exact up to ``tol``. No circular FFT wrap is involved, so arbitrarily
    large nu (slow heavy nuclei) is fine. ``collisions`` is the first state's 4 pi sigma_tot.
    """
    chain = [(coll, ip_ev, ip2_ev, yield_)] if chain is None else chain
    kernels, s_max, s_first, e = _chain_kernels(chain, dz_max, e_max, de, **kwargs)
    r_disc, d = disc_radius_cm(nc), DISC_THICKNESS_CM
    th = (np.arange(n_theta) + 0.5) * (0.5 * np.pi / n_theta)
    sg = np.pi * r_disc**2 * np.cos(th) + 2.0 * r_disc * d * np.sin(th)
    w_th = 4.0 * np.pi * sg * np.sin(th) * (0.5 * np.pi / n_theta)     # dOmega over both hemispheres
    passages = float(w_th.sum())
    if s_max <= 0.0:
        return Passages(e, de, np.zeros_like(kernels[0][0]), 0.0, passages, 0.0)
    out, beyond = _poisson_chain(kernels, s_max / sg, w_th, tol, dz_max + 1)
    return Passages(e, de, out / de, beyond, passages, 4.0 * np.pi * s_first)


def stopping_probability(coll: Collisions, nc: int, loss_ev: float, n_theta: int = 24, n_bins: int = 200,
                         tol: float = 1e-10) -> float:
    """P(the projectile loses at least ``loss_ev`` crossing the disc): its collisions ``coll`` (a distant one loses W, a
    close one ``close_loss``), Poisson-many on each chord as in ``passage_outcomes`` (nu = sigma_tot / sigma_g), over
    isotropic passages. The n-fold sums of the single losses are histograms on ``n_bins`` bins below ``loss_ev``, each loss
    at its bin's lower edge, so P is low by at most one bin per collision. An electron that does not lose its energy at
    infinity passes through: Weingartner & Draine (2001, Sect. 3.1)'s condition for a capture."""
    if loss_ev <= 0.0:
        return 1.0
    tw = _trapezoid_weights(coll.w)
    loss = np.concatenate([coll.w, coll.close_loss])
    rate = np.concatenate([coll.distant * tw, coll.close * tw])
    sigma = float(rate.sum())
    if sigma <= 0.0:
        return 0.0
    k = np.floor(loss / (loss_ev / n_bins)).astype(int)
    keep = k < n_bins
    p = np.bincount(k[keep], rate[keep], n_bins) / sigma                  # one collision, staying below loss_ev
    r, d = disc_radius_cm(nc), DISC_THICKNESS_CM
    th = (np.arange(n_theta) + 0.5) * (0.5 * np.pi / n_theta)
    sg = np.pi * r**2 * np.cos(th) + 2.0 * r * d * np.sin(th)
    nu = sigma / sg
    q, dist = [1.0], np.eye(1, n_bins).ravel()                              # P(sum of n losses < loss_ev)
    n_max = int(nu.max() + 10.0 * np.sqrt(nu.max()) + 10)
    for _ in range(n_max):
        dist = np.convolve(dist, p)[:n_bins]
        q.append(float(dist.sum()))
        if q[-1] < tol:
            break
    n = np.arange(len(q))
    pois = np.exp(n[None, :] * np.log(nu[:, None]) - nu[:, None] - gammaln(n + 1.0)[None, :])
    below = pois @ np.array(q)
    w_th = sg * np.sin(th)
    return float(1.0 - (w_th * below).sum() / w_th.sum())


def _edge_split(one: "Deposition", dz_max: int):
    """(p[dz, k], the mass beyond the grid, the grid E_k = k de, k = 0 ... n_bins): one collision's ``deposition`` with
    each bin's mass split equally between the two grid points at its edges. That keeps every collision's mean E*, so the
    n-fold sums of ``_poisson_mixture`` are exact in the mean (bin centres would lose de/2 per extra collision), and puts
    an energy of exactly 0 at k = 0."""
    n_e = one.e.size
    p = np.zeros((dz_max + 1, n_e + 1))
    lost = 0.0
    for dz, r in one.rates.items():
        p[dz, :-1] += 0.5 * r * one.de
        p[dz, 1:] += 0.5 * r * one.de
        lost += one.overflow[dz]
    return p, lost, np.arange(n_e + 1) * one.de


#: the largest negative value FFT round-off may leave in a convolution of non-negative distributions, relative to its
#: largest value, before ``_poisson_mixture`` treats it as an error (it then sets the round-off to 0)
FFT_ROUND_OFF = 1e-12


def _poisson_chain(kernels, nu, w, tol, dz_n):
    """(sum_i w_i sum_n Poisson(n; nu_i) x the outcome of n collisions, less the paths without one; the weight that
    leaves the grid) when a collision's outcome depends on the charges the passage has added so far.

    ``kernels[k]`` = (p_k[dz', E* bin], rate_k): one collision's outcome at k charges added (probabilities; the part off
    the grid is the deficit of its sum) and its rate relative to the largest (the last kernel serves every k beyond).
    ``nu``: the mean collisions of each path at the largest rate, ``w`` its weight; ``dz_n`` rows of charge are followed.
    Uniformisation: each of the Poisson(nu) events is a collision with probability rate_k at k, else nothing, so the
    result is exact. The sums go by linear convolution in E* (zero-padded FFT, no wrap), truncated to the grid; the mass
    beyond it, and the paths with more events than followed once their tail is below ``tol``, are returned apart. With
    every kernel the same this is the n-fold sum p^{*n} that served before 2026-11-05, when every collision of a passage
    took its first charge's IP and yield (pahchem audit M1)."""
    n_e = kernels[0][0].shape[1]
    nfft = 2 * n_e
    rows = max(int(np.max(np.nonzero(p.any(axis=1))[0], initial=0)) + 1 for p, _ in kernels)
    khat = [np.fft.rfft(p[:rows], n=nfft, axis=1) * rate for p, rate in kernels]
    stay = np.array([1.0 - kernels[min(k, len(kernels) - 1)][1] for k in range(dz_n)])
    cur = np.zeros((dz_n, n_e))
    cur[0, 0] = 1.0
    out, beyond = np.zeros_like(cur), 0.0
    total = float(w.sum())
    for n in range(0, 1_000_000):
        if n > 0:
            cur_hat = np.fft.rfft(cur, n=nfft, axis=1)
            acc = np.zeros((dz_n + rows, nfft // 2 + 1), dtype=complex)
            for k in np.nonzero(cur.any(axis=1))[0]:
                acc[k:k + rows] += cur_hat[k] * khat[min(k, len(khat) - 1)]
            full = np.fft.irfft(acc, n=nfft, axis=1)
            if full.min() < -FFT_ROUND_OFF * max(full.max(), 1e-300):
                raise ValueError(f"convolution negative beyond round-off: {full.min():.3e} against {full.max():.3e}")
            cur = np.maximum(full, 0.0)[:dz_n, :n_e] + stay[:, None] * cur     # collisions (round-off below 0 only), null
        w_n = float((w * np.exp(n * np.log(nu) - nu - gammaln(n + 1))).sum())       # Poisson(n; nu_i)
        in_grid = float(cur.sum())
        out += w_n * cur
        beyond += w_n * (1.0 - in_grid)
        tail = float((w * gammainc(n + 1, nu)).sum())                  # paths with more than n events
        if n > 0 and (in_grid < tol or (n > nu.max() and tail < tol * total)):
            beyond += tail                                              # they end beyond, or are negligible
            break
    out[0, 0] -= float((w * np.exp(-nu * kernels[0][1])).sum())        # the paths without a collision
    if out[0, 0] < -FFT_ROUND_OFF * max(total, 1e-300):
        raise ValueError(f"no-collision paths exceed the zero-energy outcome: {out[0, 0]:.3e}")
    out[0, 0] = max(out[0, 0], 0.0)                                     # the round-off only
    return out, beyond


def _chain_kernels(chain, dz_max, e_max, de, **kwargs):
    """(kernels for ``_poisson_chain``, the largest total cross section [cm^2], the first state's, the grid E_k): one
    collision's ``deposition`` at each state of ``chain`` = [(Collisions, IP, IP2, yield[, hole]) at 0, 1, ... charges
    added], on the grid of ``_edge_split``."""
    split = [_edge_split(deposition(c, ip, ip2, y, e_max=e_max, de=de, hole=h[0] if h else None, **kwargs), dz_max)
             for c, ip, ip2, y, *h in chain]
    sig = [float(p.sum() + lost) for p, lost, _ in split]
    s_max = max(sig)
    kernels = [(p / s if s > 0.0 else np.zeros_like(p), s / s_max if s_max > 0.0 else 0.0)
               for (p, _, _), s in zip(split, sig, strict=True)]
    return kernels, s_max, sig[0], split[0][2]


@lru_cache(maxsize=None)
def _escape_paths(nc: int, n: int = 24):
    """(weights summing to 1, path lengths [cm]) of an electron leaving the molecule from inside: Micelotta+11's disc
    (``disc_radius_cm``, DISC_THICKNESS_CM), the electron starting at a uniform point of the mid-plane (a carbon atom) and
    leaving isotropically through the faces or the rim; Gauss-Legendre in r^2, |cos theta| and phi (nodes strictly
    inside, so no path is grazing or normal exactly)."""
    x, wq = np.polynomial.legendre.leggauss(n)
    R, d = disc_radius_cm(nc), DISC_THICKNESS_CM
    r, mu, phi = R * np.sqrt(0.5 * (x + 1.0)), 0.5 * (x + 1.0), np.pi * (x + 1.0)
    rr, mm, pp = np.meshgrid(r, mu, phi, indexing="ij")
    ww = (0.5 * wq)[:, None, None] * (0.5 * wq)[None, :, None] * (0.5 * wq)[None, None, :]
    st = np.sqrt(1.0 - mm**2)
    b = rr * st * np.cos(pp)
    rim = (-b + np.sqrt(b**2 + st**2 * (R**2 - rr**2))) / st**2
    return ww.ravel(), np.minimum(rim, 0.5 * d / mm).ravel()


def escape_outcomes(coll: Collisions, nc: int, ip_ev: float, ip2_ev: float, yield_, de: float = 0.5,
                    e_max: float = 320.0, dz_max: int = 6, tol: float = 1e-12, chain=None, **kwargs):
    """What an electron leaving the molecule from inside (an outgoing photo- or Auger electron) does to it on the way:
    (P[dz, k] the probability of ending with dz more charges and internal energy E* = k ``de``, k = 0 ... e_max / de, the
    probability beyond the grid). Poisson(nu) collisions along each path of ``_escape_paths``, nu = sigma_tot L / V (a
    uniform electron density over the disc, as ``passage_outcomes``), each with ``deposition``'s outcome on the grid of
    ``_edge_split``, at the charges added so far (``chain`` as for ``passage_outcomes``); no collision sits at E* = 0.
    The collision's own mass off the grid is in the mixture's beyond (it was counted twice before 2026-11-05, pahchem
    audit m11)."""
    chain = [(coll, ip_ev, ip2_ev, yield_)] if chain is None else chain
    kernels, s_max, s_first, _ = _chain_kernels(chain, dz_max, e_max, de, **kwargs)
    w, L = _escape_paths(int(nc))
    per_sigma = L / (np.pi * disc_radius_cm(nc) ** 2 * DISC_THICKNESS_CM)
    P = np.zeros_like(kernels[0][0])
    P[0, 0] = float((w * np.exp(-s_first * per_sigma)).sum())
    if s_max <= 0.0:
        return P, 0.0
    out, beyond = _poisson_chain(kernels, s_max * per_sigma, w, tol, dz_max + 1)
    return P + out, beyond


# --- what the deposited energy does: sequential statistical fragmentation ---

@dataclass(frozen=True)
class Channel:
    """One unimolecular loss channel: barrier, activation entropy at 1000 K, path degeneracy,
    the mode removed as reaction coordinate, and the C and H atoms it takes away."""

    name: str
    e0_ev: float
    ds_j_mol_k: float
    degeneracy: float
    coordinate_cm: float
    carbon: int
    hydrogen: int


def rrkm_whitten_rabinovitch(freqs_cm, channel: Channel, energy_ev) -> np.ndarray:
    """RRKM k(E) [s^-1] with Whitten-Rabinovitch sums and densities.

    It is usable far above the range of a Beyer-Swinehart count, which
    overflows float64 near 100 eV for N_C ~ 50. The transition state is
    ``rates.transition_state_for_entropy`` (n_scaled = 5, West+18).
    """
    from pycalima.pahchem import dos, rates

    E = np.asarray(energy_ev, dtype=float) * dos.CM_PER_EV
    ts = rates.transition_state_for_entropy(freqs_cm, channel.coordinate_cm, channel.ds_j_mol_k, n_scaled=5)
    e0 = channel.e0_ev * dos.CM_PER_EV
    _, log_rho = dos.whitten_rabinovitch(freqs_cm, E)
    above = E > e0
    log_n = np.full_like(E, -np.inf)
    log_n[above] = dos.whitten_rabinovitch(ts, E[above] - e0)[0]
    return np.where(above, channel.degeneracy * rates.C_CM_S * np.exp(log_n - log_rho), 0.0)


class FragmentationChain:
    """Expected atoms lost after a deposit of internal energy E*.

    Above the dissociation onset, a hot PAH loses one fragment after another,
    each loss chosen with branching k_c / sum k. Each loss removes its barrier
    plus a kinetic release ~ k T_micro (Klots), until the remainder cools by
    IR emission first. ``survival(E)`` is the probability of no loss before
    cooling. It comes from the IR cascade (``energy_me.CascadeModel.photon_yields``
    with the same k_c) up to ``e_cascade_ev`` and is 0 above. The recursion

        X(E) = [1 - survival(E)] sum_c b_c(E) [x_c + X(E - E0_c - k T(E))]

    is solved upward on a grid. It gives C(E) and H(E), the expected carbon
    and hydrogen atoms lost, and ``radiated``: the internal energy R(E) the
    last daughter keeps and radiates in the IR,

        R(E) = survival(E) E + [1 - survival(E)] sum_c b_c(E) R(E - E0_c - k T(E)).

    Above the onset R(E) saturates near the onset energy, about 16-17 eV for
    C54H18 (the evaporative-ensemble ceiling). Photons emitted on the way down,
    before each loss, are neglected: k_diss >> k_IR there. A mixture of
    outcomes at one E is represented by its mean residual energy. The daughters
    keep the parent's modes and rates, an assumption that matters once many H
    are gone.
    """

    def __init__(self, species, channels, e_max_ev: float = 512.0, de_ev: float = 0.05,
                 e_cascade_ev: float = 40.0, de_cm: float = 100.0):
        from pycalima.pahchem import dos, energy_me

        self.channels = tuple(channels)
        self.e = np.arange(0.0, e_max_ev + 0.5 * de_ev, de_ev)
        freqs = species.freqs_cm
        k = np.array([rrkm_whitten_rabinovitch(freqs, c, self.e) for c in self.channels])
        k_tot = k.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.branching = np.where(k_tot > 0, k / k_tot, 0.0)
        cm = energy_me.CascadeModel(freqs, species.einstein_a, e_max_cm=e_cascade_ev * dos.CM_PER_EV, de_cm=de_cm)
        ec = cm.energy_cm / dos.CM_PER_EV
        y = cm.photon_yields({c.name: np.interp(ec, self.e, kc) for c, kc in zip(self.channels, k)})
        p_diss = np.clip(sum(y.values()), 0.0, 1.0)
        self.p_diss = np.where(self.e <= ec[-1], np.interp(self.e, ec, p_diss), 1.0)
        _, log_rho = dos.whitten_rabinovitch(freqs, np.maximum(self.e, 1e-3) * dos.CM_PER_EV)
        with np.errstate(divide="ignore", invalid="ignore"):
            kt_ev = 1.0 / (np.gradient(log_rho, self.e * dos.CM_PER_EV) * dos.CM_PER_EV)
        self.kt_ev = np.nan_to_num(kt_ev, nan=0.0, posinf=0.0, neginf=0.0)
        self.carbon, self.hydrogen, self.radiated = self._solve()

    def _solve(self):
        n = self.e.size
        de = self.e[1] - self.e[0]
        out = np.zeros((3, n))
        x = np.array([[c.carbon for c in self.channels], [c.hydrogen for c in self.channels],
                      [0.0 for _ in self.channels]], dtype=float)
        e0 = np.array([c.e0_ev for c in self.channels])
        for i in range(n):
            out[2, i] = (1.0 - self.p_diss[i]) * self.e[i]
            if self.p_diss[i] <= 0.0:
                continue
            rest = self.e[i] - e0 - self.kt_ev[i]
            j = np.clip(np.floor(rest / de).astype(int), 0, i - 1 if i > 0 else 0)
            frac = np.clip(rest / de - j, 0.0, 1.0)
            later = out[:, j] * (1.0 - frac) + out[:, np.minimum(j + 1, i - 1 if i > 0 else 0)] * frac
            later = np.where(rest >= 0.0, later, 0.0)
            out[:, i] += self.p_diss[i] * ((x + later) * self.branching[:, i]).sum(axis=1)
        return out[0], out[1], out[2]

    def lost(self, energy_ev):
        """(carbon, hydrogen) expected loss after a deposit E* [eV]."""
        E = np.asarray(energy_ev, dtype=float)
        return np.interp(E, self.e, self.carbon), np.interp(E, self.e, self.hydrogen)

    def residual(self, energy_ev):
        """Internal energy radiated by the last daughter after a deposit E* [eV]."""
        return np.interp(np.asarray(energy_ev, dtype=float), self.e, self.radiated)


#: The highest charge a cold PAH holds, per carbon: floor(0.2 N_C) gives the thermodynamic limits of Holm+11
#: (DFT; q = 2, 2, 2, 3, 4 for naphthalene, biphenylene, anthracene, pyrene, coronene) and the C60^12+ seen intact
#: (14 computed, Diaz-Tendero+05). Metastable ions reach one more at C14-C16 (PAH^3+, Lee+22; pyrene^4+, Lawicki+11),
#: ovalene^3+ and HBC^3+ are seen intact (Zhen+15). Beyond C60 an extrapolation; confidence C.
COLD_CHARGE_PER_C = 0.2
#: Chabot+20 Fig. 7 (C60, after Qian+13 and Martin+08): the internal energy [eV] at which multifragmentation and fission
#: are equally probable, 105 eV at q = 3 and 60 eV at q = 5 (read from the figure, +-5 eV). Chabot+20 scale it to every
#: size as Q/N_C and E*/N_C; along their CR internal energy (~20 eV per charge) it crosses at Q/N_C ~ 4/60, the
#: threshold ``COULOMB_CHARGE_PER_C``, which is therefore a hot-ion boundary, not a cold limit. Confidence D.
MULTIFRAGMENTATION_C60 = ((3, 105.0), (5, 60.0))
#: Chabot+20's threshold as they apply it (``coulomb="charge_only"``), kept for comparison
COULOMB_CHARGE_PER_C = 4.0 / 60.0


#: (a, b) of the fission barrier for proton emission, B = a - b q / sqrt(N_C) [eV]: Holm+11 (J. Chem. Phys. 134, 044301)
#: Fig. 7, B3LYP/6-31G(d) for naphthalene, biphenylene, anthracene, pyrene and coronene at q = 3-10 (23 points read from
#: the figure, +-0.1-0.2 eV; rms 0.28 eV about the fit, against 0.53 for q / N_C). q / sqrt(N_C) is the charge over the
#: disc's radius: the Coulomb energy that lowers the barrier. Confidence C (DFT, 5 molecules, N_C <= 24)
FISSION_BARRIER = (6.30, 2.88)


def fission_barrier(nc: int, charge) -> np.ndarray:
    """B(q) [eV] of proton emission from a PAH of ``nc`` carbons at charge q (``FISSION_BARRIER``)."""
    a, b = FISSION_BARRIER
    return a - b * np.asarray(charge, dtype=float) / np.sqrt(nc)


def coulomb_limit(nc: int) -> int:
    """The highest charge a cold PAH of ``nc`` carbons survives: floor(``COLD_CHARGE_PER_C`` N_C), or the highest charge
    whose proton-emission barrier is still positive (``fission_barrier``, q < (a / b) sqrt(N_C)) where that is lower (the
    user's choice, 2026-11-04). The two agree up to N_C ~ 115; above it the barrier limits: C150 26 (not 30), C216 32 (not
    43). Both the thermodynamic and the barrier limits scale with the radius, sqrt(N_C). No astrophysical model prescribes
    either: PAH codes truncate (Murga+22 -2 ... +8, Montillaud+13 0, +1); grain physics (WDB06 Sect. 10.2.1, Hoang+15:
    Coulomb explosion or ion field emission) scales as a^2 ~ N_C^(2/3), C216 ~12-14, but puts coronene and C60 at ~3,
    against coronene4+ and C60^12+ seen intact, so for molecules it is a lower bound."""
    a, b = FISSION_BARRIER
    return int(min(np.floor(COLD_CHARGE_PER_C * nc), np.ceil(a / b * np.sqrt(nc)) - 1))


def multifragmentation_energy(nc: int, charge) -> np.ndarray:
    """E* [eV] above which an ion of ``charge`` multifragments (``MULTIFRAGMENTATION_C60`` per carbon); inf below
    Q/N_C = 3/60, where Chabot+20's figure shows none, and held at its q = 5 value above Q/N_C = 5/60."""
    (q0, e0), (q1, e1) = MULTIFRAGMENTATION_C60
    x = np.clip(np.asarray(charge, dtype=float) / nc * 60.0, q0, q1)
    e = nc * (e0 + (e1 - e0) * (x - q0) / (q1 - q0)) / 60.0
    return np.where(np.asarray(charge, dtype=float) / nc * 60.0 < q0, np.inf, e)


def passage_spectrum(nc: int, nh: int, charge: int, ip_ev: float, ip2_ev: float, spectra, yield_,
                     n_t: int = 48, w=None, t_grid=None, focus: bool = False, ladder=None, **kwargs) -> dict:
    """{projectile name: Passages} with rates per molecule: P [s^-1 eV^-1], and beyond,
    passages and collisions [s^-1], from int j(T) Passages(T) dT.

    ``spectra``: [(Projectile, j(T))] as for ``impact_rates``; ``kwargs`` go to
    ``passage_outcomes``. ``t_grid``: {projectile name: kinetic energies [eV]} instead of
    the cosmic rays' (electrons 1 keV-1 PeV, nuclei from T_MIN_PER_U_EV per nucleon).
    ``focus``: the Coulomb focusing of thermal particles on a charged PAH (``coulomb_focus``).
    ``ladder`` = [(IP, IP2, yield[, hole]) with 0, 1, ... charges added]: the collisions after k ionisations are on
    charge ``charge`` + k's oscillator strengths, IP and yield (``passage_outcomes``' chain); None: all at ``charge``.
    """
    w = w_grid() if w is None else w
    ladder = [(ip_ev, ip2_ev, yield_)] if ladder is None else ladder
    dfs = [oscillator_strength(nc, nh, charge + k, w) for k in range(len(ladder))]
    out = {}
    for proj, j in spectra:
        if t_grid is not None and proj.name in t_grid:
            t = np.asarray(t_grid[proj.name], dtype=float)
        elif proj.is_electron:
            t = np.geomspace(1.0e3, 1.0e12, n_t)
        else:
            t = np.geomspace(T_MIN_PER_U_EV * proj.nucleons, 1.0e11 * proj.nucleons, n_t)
        g, t_hit = coulomb_focus(proj, charge, nc, t) if focus else (np.ones_like(t), t)
        wt = _trapezoid_weights(np.log(t)) * t * np.asarray(j(t), dtype=float) * g
        acc = None
        for ti, wi in zip(t_hit, wt):
            if wi <= 0.0:
                continue
            corr = low_speed_correction(proj, ti)
            chain = [(collisions(proj, ti, df, w, correction=corr, ip_ev=ip), ip, ip2, y, *h)
                     for df, (ip, ip2, y, *h) in zip(dfs, ladder, strict=True)]
            ps = passage_outcomes(chain[0][0], nc, ip_ev, ip2_ev, yield_, chain=chain, **kwargs)
            ps = Passages(ps.e, ps.de, wi * ps.P, wi * ps.beyond, wi * ps.passages, wi * ps.collisions)
            acc = ps if acc is None else Passages(ps.e, ps.de, acc.P + ps.P, acc.beyond + ps.beyond,
                                                  acc.passages + ps.passages, acc.collisions + ps.collisions)
        out[proj.name] = acc
    return out


def maxwellian_flux(proj: Projectile, temperature_k: float, density: float):
    """j(T) [particles eV^-1 s^-1 cm^-2 sr^-1] of a thermal gas of ``proj`` at ``temperature_k`` and ``density`` [cm^-3]:
    n v(T) f(T) / 4 pi, f the Maxwell-Boltzmann distribution of kinetic energies, at non-relativistic speeds (kT <~ 10 keV)."""
    kt = 8.617333262e-5 * float(temperature_k)

    def j(t_ev):
        t = np.asarray(t_ev, dtype=float)
        f = 2.0 / np.sqrt(np.pi) * kt ** -1.5 * np.sqrt(t) * np.exp(-t / kt)
        return float(density) * C_CM_S * np.sqrt(2.0 * t / proj.rest_ev) * f / (4.0 * np.pi)
    return j


def coulomb_focus(proj: Projectile, charge: int, nc: int, t_ev):
    """(g, T at impact) of a projectile reaching a PAH of ``charge``: classically, through the potential U = q Z e^2 / R on
    Micelotta's disc (R = 0.9 sqrt(N_C) A; q = -1 for electrons, z for nuclei), the passage rate scales by g = 1 - U / T
    (> 1 attracted, 0 below a repulsive barrier) and the particle arrives with T - U. Matters for thermal particles only:
    U = 2.7 eV for an electron on C24+, T ~ 8.6 eV at 1e5 K."""
    t = np.asarray(t_ev, dtype=float)
    u = (-1.0 if proj.is_electron else float(proj.z)) * float(charge) * E2_EV_CM / disc_radius_cm(nc)
    return np.clip(1.0 - u / t, 0.0, None), np.maximum(t - u, 0.0)


# --- thermal ions: electronic friction and elastic collisions (Micelotta, Jones & Tielens 2010a, b) -------------------

BOHR_CM = 0.529177210903e-8
HARTREE_EV = 27.211386245988
#: the atomic unit of speed, alpha c [cm s^-1]
V_AU_CM_S = 2.18769126364e8
#: (n_0, w) of the disc's valence-electron density n_0 exp(-x^2 / w) [a.u.], x the height above its plane [a.u.]
#: (Micelotta+10b eq. 4, after Hadjar+01 for C60). Its column, 0.437 a.u. = 1.56 e A^-2, is graphene's four valence
#: electrons per C (1.53 e A^-2)
JELLIUM = (0.15, 2.7)
#: the height [a.u.] beyond which the disc holds nothing: the H friction is below 1e-6 of its peak beyond 5.2 a.u.
JELLIUM_EXTENT_AU = 8.0
#: the friction of a slow atom in an electron gas, eta(Z, r_s) [a.u.], is interpolated in Gerrits, Juaristi & Meyer
#: (2020, PRB 102, 155130)'s LDA tables (``gerrits2020_friction_lda.csv``, Z = 1-30 at r_s = 1.5-5), which reproduce
#: Puska & Nieminen (1983, PRB 27, 6121) and so the friction Micelotta+10b fit, gamma = Gamma_0 exp(-(r_s - 1.5) / R_2)
#: (their eq. 6). Until 2026-11-05 that fit served, with PAH_sputtering's unsourced Gamma_0 (H 0.33 against Puska &
#: Nieminen's 0.310) and for H, He, C and O only.
FRICTION_FILE = "gerrits2020_friction_lda.csv"
#: {element: (Z, mass [amu])}: the thermal ions of ``chemistry.IONS`` and the PAH's atoms
ATOMS = {"H": (1, 1.008), "He": (2, 4.0026), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999), "Ne": (10, 20.180),
         "Na": (11, 22.990), "Mg": (12, 24.305), "Si": (14, 28.086), "S": (16, 32.06), "Fe": (26, 55.845)}
#: {target: the energy [eV] an elastic collision must give the PAH's atom to remove it}. C: Stockett+15 (J. Phys. Chem.
#: Lett. 6, 4504), the MD energy transfer at their measured He + PAH+ thresholds (anthracene, pyrene, coronene),
#: 23.3 +- 0.3 eV (their MD alone 27.0; Postma+14 MD of anthracene 27; graphene 23.6); Micelotta+10a (MJT) took 7.5
#: (4.5-15) eV. H: 9 eV, Chen+14 (J. Chem. Phys. 140, 224306) after Postma+14's MD (not checked in Postma's tables)
KNOCKOUT_T0_EV = {"C": 23.3, "H": 9.0}
#: Ziegler+85's exponent m(epsilon) of the power-law cross section (MJT eq. 13) is fitted from epsilon_1 = 1e-9 up
ZIEGLER_EPS1 = 1.0e-9
ZIEGLER_M = (-2.432, -0.1509, 2.648, -2.742, 1.215, -0.1665)


@lru_cache(maxsize=None)
def _friction_table() -> dict:
    """{element: (r_s [a.u.], eta [a.u.])} of FRICTION_FILE, the tabulated points only."""
    with open(_paths.get_external_data_path("pahchem", FRICTION_FILE)) as f:
        lines = [line.strip() for line in f if not line.startswith("#")]
    cols = lines[0].split(",")
    rs = np.array([float(c[2:]) for c in cols[2:]])
    out = {}
    for line in lines[1:]:
        v = line.split(",")
        eta = np.array([float(x) if x else np.nan for x in v[2:]])
        ok = np.isfinite(eta)
        out[v[1]] = (rs[ok], eta[ok])
    return out


def has_friction(element: str | None) -> bool:
    return element is not None and element in _friction_table()


def friction_coefficient(element: str, x_au) -> np.ndarray:
    """gamma [a.u.] of a slow ion of ``element`` at height ``x_au`` [a.u.] above the disc (``JELLIUM``): ln eta
    interpolated linearly in r_s between the tabulated points of FRICTION_FILE, continued beyond them with the slope of
    the outer two (denser than r_s = 1.5, the disc's core at 1.17, and above 5, where it is negligible)."""
    n0, w = JELLIUM
    rs_t, eta_t = _friction_table()[element]
    n = n0 * np.exp(-np.asarray(x_au, dtype=float) ** 2 / w)
    with np.errstate(divide="ignore", over="ignore"):
        rs = (4.0 / 3.0 * np.pi * n) ** (-1.0 / 3.0)
    le = np.log(eta_t)
    lo, hi = (le[1] - le[0]) / (rs_t[1] - rs_t[0]), (le[-1] - le[-2]) / (rs_t[-1] - rs_t[-2])
    with np.errstate(invalid="ignore", over="ignore"):
        ln = np.where(rs < rs_t[0], le[0] + lo * (rs - rs_t[0]),
                      np.where(rs > rs_t[-1], le[-1] + hi * (rs - rs_t[-1]), np.interp(rs, rs_t, le)))
    return np.where(np.isfinite(rs), np.exp(ln), 0.0)


@dataclass(frozen=True)
class Chords:
    """Straight passages through the disc from an isotropic intensity. Per chord: ``friction`` = HARTREE int gamma ds
    [eV per a.u. of speed; None without a friction], ``atoms`` = the C atoms it crosses per unit area [cm^-2], and
    ``weight`` [cm^2 sr], whose sum is the passages per unit intensity."""

    friction: np.ndarray | None
    atoms: np.ndarray
    weight: np.ndarray


def disc_chords(nc: int, element: str | None = None, n_theta: int = 24, n_u: int = 96, n_w: int = 32) -> Chords:
    """Every chord through Micelotta's disc: radius R = ``disc_radius_cm``, its electrons (``JELLIUM``) and atoms spread
    over the height x alike, nothing beyond R. A chord at angle theta to the normal, offset (u, w) across the beam,
    is at height x = -u / sin theta + s cos theta along its length s, inside while its radius is below R: its columns
    are differences of the height integrals over cos theta. The mean over chords is exact (int gamma dV and N_C, to 1e-3
    on this grid); Micelotta+10b eq. 7 takes the central chord at each angle (2R edge-on, against pi R / 2 on average)."""
    r = disc_radius_cm(nc) / BOHR_CM
    x = np.linspace(0.0, JELLIUM_EXTENT_AU, 4001)

    def odd_integral(y):
        c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
        return lambda z: np.sign(z) * np.interp(np.abs(z), x, c)
    profile = np.exp(-x**2 / JELLIUM[1])
    A = odd_integral(profile / (2.0 * np.trapezoid(profile, x)))
    F = odd_integral(friction_coefficient(element, x)) if has_friction(element) else None
    dth = 0.5 * np.pi / n_theta
    fr, at, wt = [], [], []
    for t in (np.arange(n_theta) + 0.5) * dth:
        span = r * np.cos(t) + JELLIUM_EXTENT_AU * np.sin(t)
        u, w = np.meshgrid((np.arange(n_u) + 0.5) * span / n_u, (np.arange(n_w) + 0.5) * r / n_w, indexing="ij")
        mid, half = -u / np.sin(t), np.sqrt(r**2 - w**2) / np.tan(t)
        at.append((nc / (np.pi * (r * BOHR_CM) ** 2) * (A(mid + half) - A(mid - half)) / np.cos(t)).ravel())
        if F is not None:
            fr.append((HARTREE_EV * (F(mid + half) - F(mid - half)) / np.cos(t)).ravel())
        # the four quadrants of (u, w), both hemispheres
        wt.append(np.full(u.size, 4.0 * span / n_u * r / n_w * BOHR_CM**2 * 4.0 * np.pi * np.sin(t) * dth))
    return Chords(np.concatenate(fr) if F is not None else None, np.concatenate(at), np.concatenate(wt))


def knock_on(element: str, t_ev, t0_ev: float, target: str = "C"):
    """(sigma [cm^2], S_below [eV cm^2]) per ``target`` atom at rest of an elastic collision with an ion of ``element``
    and kinetic energy ``t_ev``: the cross section for transferring more than ``t0_ev`` (MJT eq. 15, zero below
    E_0n = t0 / gamma) and the stopping cross section of the transfers below it, S_n0 min(1, t0 / T_m)^(1 - m) (MJT
    eq. 14's complement), heat left in the molecule. ZBL screening length and universal reduced stopping (MJT eqs 1, 2, 5,
    6); m(epsilon) of Ziegler+85 (eq. 13) at epsilon >= ZIEGLER_EPS1, the fit's lower end (energies below 1e-4 eV)."""
    z1, m1 = ATOMS[element]
    z2, m2 = ATOMS[target]
    e = np.asarray(t_ev, dtype=float)
    e2 = E2_EV_CM * 1e8                                                       # e^2 [eV A]
    a = 0.8854 * BOHR_CM * 1e8 / (z1**0.23 + z2**0.23)                        # [A]
    eps = m2 / (m1 + m2) * a / (z1 * z2 * e2) * e
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = np.where(eps <= 30.0,
                      0.5 * np.log1p(1.1383 * eps) / (eps + 0.01321 * eps**0.21226 + 0.19593 * eps**0.5),
                      np.log(np.maximum(eps, 30.0)) / (2.0 * np.maximum(eps, 30.0)))
    sn = np.where(eps > 0.0, sn, 0.0)
    u = 0.1 * np.log(np.maximum(eps, ZIEGLER_EPS1) / ZIEGLER_EPS1)
    m = 1.0 - np.exp(-np.exp(sum(c * u**i for i, c in enumerate(ZIEGLER_M))))
    s0 = 4.0 * np.pi * a * z1 * z2 * e2 * m1 / (m1 + m2) * sn * 1e-16        # [eV cm^2]
    tm = 4.0 * m1 * m2 / (m1 + m2) ** 2 * e
    above = tm > t0_ev
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(above, s0 * (1.0 - m) / (m * tm) * ((tm / t0_ev) ** m - 1.0), 0.0)
        below = np.where(above, s0 * (t0_ev / tm) ** (1.0 - m), s0)
    return sigma, below


def thermal_ion_passages(nc: int, element: str, temperature_k: float, density: float = 1.0, charge: int = 0,
                         ion_charge: int = 1, nh: int = 0, n_t: int = 48, de: float = 1.0, e_max: float = 512.0,
                         dz_max: int = 15, t0_ev: dict = KNOCKOUT_T0_EV):
    """(Passages, {target: knock-outs [s^-1]}) per molecule of a thermal gas of ions of ``element`` and charge
    ``ion_charge`` at ``temperature_k`` and ``density`` [cm^-3] on C_{nc}H_{nh} of ``charge``.

    - **Heat** (dz = 0 only): along its chord (``disc_chords``) the ion loses v HARTREE int gamma ds to the valence
      electrons (Micelotta+10b eqs 1-7, with Gerrits+20's tabulated friction: linear in the speed, for v <~ 1 a.u.,
      H below ~1e8 K) and, to each
      target atom it crosses, S_below of elastic collisions below the target's knock-out threshold (``knock_on``). MJT
      leave the latter out ("radiated away"); it is heat like the friction's, and exceeds it below ~1e6 K for He and C.
      The H atoms, on the rim, are spread over the disc like the C atoms. Atoms without a friction table
      (``has_friction``; Z = 1-30 have one) deposit the elastic part only.
    - **Knock-outs** of each target in ``t0_ev`` (C, H): N int sigma v f(v) dv (MJT eq. 22 without its factor 1/2, their
      Appendix C's <cos theta> of the projected area: a per-atom cross section already counts every orientation, the
      atom column an ion crosses growing as 1 / cos theta where the projection shrinks as cos theta; no later work uses
      it). The molecule's shadowing of its own atoms is not followed: Chen+14's Monte Carlo puts whole-molecule cross
      sections at 0.44 (H) and 0.74 (He) of the atoms' sum, for their own per-atom cross section.

    Maxwellian fluxes (``maxwellian_flux``) Coulomb-focused on the charged PAH (``coulomb_focus``). ``P[dz, k]`` and
    ``e`` are as for ``passage_outcomes``: each chord's deposit is split between the grid points E_k = k de about it,
    keeping its mean, and those beyond ``e_max`` go to ``beyond``.
    """
    proj = nucleus(element, ion_charge, ATOMS[element][1])
    kt = 8.617333262e-5 * float(temperature_k)
    j = maxwellian_flux(proj, temperature_k, density)
    ch = disc_chords(nc, element)
    counts = {"C": nc, "H": nh}
    targets = [x for x in t0_ev if counts[x] > 0]
    n_bins = int(round(e_max / de))
    e = np.arange(n_bins + 1) * de
    P = np.zeros((dz_max + 1, n_bins + 1))
    beyond = passages = 0.0
    t = np.geomspace(1e-2 * kt, 40.0 * kt, n_t)
    g, t_hit = coulomb_focus(proj, charge, nc, t)
    wt = _trapezoid_weights(np.log(t)) * t * j(t) * g
    # the elastic stopping below threshold per C atom of the column: the targets' per-atom values, by their numbers
    below = sum((knock_on(element, t_hit, t0_ev[x], x)[1] * counts[x] / nc for x in targets), np.zeros_like(t))
    for ti, wi, bi in zip(t_hit, wt, below):
        if wi <= 0.0:
            continue
        dep = bi * ch.atoms
        if ch.friction is not None:
            dep = dep + np.sqrt(2.0 * ti / proj.rest_ev) * C_CM_S / V_AU_CM_S * ch.friction
        r = wi * ch.weight
        passages += float(r.sum())
        k = np.floor(dep / de).astype(int)
        inside = k < n_bins
        beyond += float(r[~inside].sum())
        frac = dep[inside] / de - k[inside]
        P[0] += np.bincount(k[inside], r[inside] * (1.0 - frac), n_bins + 1)
        P[0] += np.bincount(k[inside] + 1, r[inside] * frac, n_bins + 1)
    # knock-outs above the threshold energy in the gas, E_0n + U, on a grid of their own per target
    u_ev = float(proj.z) * float(charge) * E2_EV_CM / disc_radius_cm(nc)
    knockouts = {}
    for x in targets:
        m1, m2 = ATOMS[element][1], ATOMS[x][1]
        e0 = max(t0_ev[x] * (m1 + m2) ** 2 / (4.0 * m1 * m2) + u_ev, 0.0)
        tk = e0 + kt * np.concatenate([[0.0], np.geomspace(1e-4, 40.0, 95)])
        gk, tk_hit = coulomb_focus(proj, charge, nc, tk)
        sigma, _ = knock_on(element, tk_hit, t0_ev[x], x)
        knockouts[x] = float(counts[x] * np.sum(_trapezoid_weights(tk) * 4.0 * np.pi * j(tk) * gk * sigma))
    return Passages(e, de, P / de, beyond, passages, passages), knockouts


def destruction_rates(spectrum: dict, nc: int, nh: int, charge: int, chain: FragmentationChain,
                      coulomb: str = "cold_and_hot") -> dict:
    """Per projectile: the rates [s^-1 per molecule] of what passages do to C_{nc}H_{nh}^Z.

    ``spectrum`` is a ``passage_spectrum`` result. The keys are:

    - ``collisions``, ``passages``: rates of collisions and of crossings.
    - ``ionised``: passages ending at charge Z + dz with dz >= 1.
    - ``heat``: internal energy deposited [eV s^-1], counting in-grid passages only.
    - ``carbon``, ``hydrogen``: atoms lost per second by statistical fragmentation
      (``chain``), from passages that neither explode nor go beyond the grid.
    - ``exploded``: passages that end in Coulomb explosion. With ``coulomb="cold_and_hot"`` (default) those are
      the ones ending above the cold limit ``coulomb_limit(nc)`` whatever their E*, and those whose E* exceeds
      ``multifragmentation_energy`` at their final charge. ``"charge_only"`` is Chabot+20's reading, charge >=
      ``COULOMB_CHARGE_PER_C`` x N_C at any E* (it makes cold C24H12 dications explode; lab ones are stable).
    - ``beyond``: passages with E* beyond the grid.
    - ``destroyed``: exploded + beyond + those whose statistical loss reaches
      N_C / 3 (Micelotta+11's criterion).
    """
    if coulomb not in ("cold_and_hot", "charge_only"):
        raise ValueError(f"coulomb must be 'cold_and_hot' or 'charge_only', not {coulomb!r}")
    q_max = coulomb_limit(nc) if coulomb == "cold_and_hot" else int(np.ceil(COULOMB_CHARGE_PER_C * nc)) - 1
    if charge > q_max:
        raise ValueError(f"charge {charge} is already above the highest stable charge {q_max}")
    out = {}
    for name, ps in spectrum.items():
        P = ps.P * ps.de
        q = charge + np.arange(P.shape[0])[:, None]
        boom = q > q_max
        if coulomb == "cold_and_hot":
            boom = boom | (ps.e[None, :] >= multifragmentation_energy(nc, q))
        Cl, Hl = chain.lost(ps.e)
        gone = (Cl >= nc / 3.0)[None, :]
        keep = np.where(boom, 0.0, P)
        exploded = float(np.where(boom, P, 0.0).sum())
        out[name] = {"collisions": ps.collisions, "passages": ps.passages, "ionised": float(P[1:].sum()),
                     "heat": float((P.sum(axis=0) * ps.e).sum()),
                     "carbon": float((keep * np.where(gone, 0.0, Cl[None, :])).sum()),
                     "hydrogen": float((keep * np.where(gone, 0.0, np.minimum(Hl, nh)[None, :])).sum()),
                     "exploded": exploded, "beyond": ps.beyond,
                     "destroyed": exploded + ps.beyond + float((keep * gone).sum())}
    return out


def lifetime_yr(rates: dict, nc: int) -> float:
    """1 / (destroyed + carbon / (N_C / 3)) in years: Micelotta's 1/3-of-the-carbon criterion,
    applied to the gradual loss, plus the passages that destroy outright."""
    r = sum(v["destroyed"] + v["carbon"] / (nc / 3.0) for v in rates.values())
    return float(1.0 / r / 3.15576e7) if r > 0 else np.inf
