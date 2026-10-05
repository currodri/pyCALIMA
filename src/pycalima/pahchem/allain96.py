"""Allain, Leach & Sedlmayr (1996, papers I and II) run through pahchem.

The inputs come from ``method_cards/allain96.toml``. The model, as the card
reads it:

- classical RRK, k_i(E) = k0 (1 - E0/E)^(3N - 7), N = N_C + N_H, one
  (k0, E0) per channel for every PAH, neutral or cation, hydrogenated or not
  (I Table 1, from Jochims+94). The H2 and C2H2 channels share E0, so
  k_C2H2/k_H2 = 0.1 at every E;
- first-event branching at E = h nu, with no cascade and no multiphoton:
  Y_i = (1 - Y_ion) k_i/(k_IR + k_diss) (I eqs. 16-17), and per-molecule
  rates int sigma N(E) Y_i dE over 2.8-13.6 eV (I eq. 20);
- k_IR(E) = sum_i A_i <n_i>(T*), with T*(E) from the harmonic U(T) (I eqs.
  8-10). Allain drop the occupation factor for ovalene and larger, where
  k_IR = sum_i A_i;
- the I eq. 11 ionisation yield, the Table 2 IPs or the eq. 12 IP law;
- in paper II, a (Z = 0, +1) x (N_H = 0..N_H0) steady state (II eqs. 13-16).
  It has Spitzer recombination (eq. 9), constant H addition, benzene-like
  abstraction (eq. 12), H+ reactions, and C loss evaluated afterwards
  (eq. 24).

Where the card says "not stated", or where pahchem lacks the paper's input,
this module chooses as follows. Every choice is confidence E.

- **Cross sections.** Joblin (1992) is not available. Instead:
  - Malloci+07 TD-DFT spectra (``allain96_sigma_malloci2007.csv``) for
    naphthalene, anthracene, pyrene, coronene and ovalene, neutral and
    cation;
  - Li & Draine (2001) for the 50 and 100 C PAHs (Allain used Verstraete &
    Leger 1992 there);
  - for benzene, no gas-phase sigma is bundled, so it enters only the
    analytic gates.
- **Einstein A.** Allain's Leger+89b band A values are not stated. PAHdb v4
  modes are used, and cations take the neutral's modes (as in II Sect. 2.1).
- **Dehydrogenated states.** The card says the C-H bands are dropped and the
  other modes kept. Here, per missing H, the most intense remaining C-H
  stretch, in-plane bend and out-of-plane bend are removed; a band that runs
  out of modes stops losing them.
- **"PAH 50 C" and "PAH 100 C".** N_H0 is not stated. They are PAHdb C50H20
  (uid 10508, C2h) and C100H26 (uid 7495, C2h).
- **Set 1, Draine 1978, below 5 eV.** This is 4 pi times I eq. 19, taken to
  be per steradian like eq. 18; eq. 18 is used from 5 eV up. The two do not
  join at 5 eV (factor ~2; card [field].set1_check).
- **Set 2, Gondhalekar+80** (paper II's only field). Read from
  ``SET2_FILE``, digitised from I Fig. 5 (2.88-13.59 eV). ``set2_proxy``
  (Set 1 / 1.6, the card's set_ratio) remains as a fallback.
- **k_IR for ovalene and larger.** sum_i A_i, the card's literal reading of
  "neglected". With the occupation factor kept, the C32H14 / PAH 50 C H-loss
  rates rise 2.6x / 10x (check_allain96.py).
- **Recombination.** II eq. 9 with e^2/a, P = 0.3 and x0.8. This reproduces
  their benzene+ 1e-6 cm^3 s^-1 at 300 K; eq. 10's printed prefactor is 3.3x
  lower.
- **n(H+).** Not tabulated for the six media. It is taken as
  max(n_e - 5e-4 n_H, 0) from II eq. 22, and is zero in four of the media.
- **Electronic luminescence** (S1/T1 fluorescence, phosphorescence, IIC) is
  left out; the card says it matters only below ~6 eV for N_C <= 24.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import cooling, dos, fields, photo, rates
from pycalima.pahchem.molecule import PAHSpecies
from pycalima.pahchem.network import PAHNetwork

YR = 3.15576e7
E_GRID = np.round(np.arange(2.8, 13.6 + 1e-9, 0.02), 6)        # photon energies [eV]

#: I Table 1 (Jochims+94 RRK fit): channel -> (k0 [s^-1], E0 [eV])
CHANNELS = {"H": (1e16, 2.8), "H2": (1e16, 2.9), "C2H2": (1e15, 2.9)}
#: Allain's species -> (N_C, N_H0, Malloci name or None, PAHdb mode file or None)
SPECIES = {
    "C6H6": (6, 6, None, None),
    "C10H8": (10, 8, "naphthalene", "C10H8_0.dat"),
    "C14H10": (14, 10, "anthracene", "C14H10_0.dat"),
    "C16H10": (16, 10, "pyrene", "C16H10_0.dat"),
    "C24H12": (24, 12, "coronene", "C24H12_0.dat"),
    "C32H14": (32, 14, "ovalene", "C32H14_0.dat"),
    "PAH 50 C": (50, 20, None, "C50H20_0.dat"),
    "PAH 100 C": (100, 26, None, "C100H26_0.dat"),
}
#: I Table 2 ionisation energies [eV]; others from I eq. 12
TABLE2_IP = {"C6H6": 9.24, "C10H8": 8.15, "C14H10": 7.41, "C16H10": 7.41, "C24H12": 7.29}
SET_RATIO = 1.6                    # Set 1 / Set 2 rates (I Sect. 3.2)
MULTI_QUANTA_MAX_NC = 24           # occupation factor kept below ovalene (I Sect. 2.3)

#: II Table 4: name -> (n_H, n_e [cm^-3], T [K], chi)
MEDIA = {
    "NGC 7027": (7e4, 7e1, 5e2, 4.6e5),
    "Orion Bar": (1e5, 3e1, 5e2, 4.3e3),
    "NGC 7023": (4e3, 1e0, 2.5e2, 2.5e3),
    "M82": (1e4, 3e0, 2.5e2, 2.9e3),
    "HI region": (5e1, 2e-2, 8e1, 1.0),
    "diffuse warm medium": (5e-1, 4e-3, 8e3, 1.0),
}
X_C, K_C = 3e-4, 1e-12             # carbon abundance, C accretion [cm^3 s^-1] (I eq. 26)


# ---------------------------------------------------------------------------
# Analytic pieces
# ---------------------------------------------------------------------------

def rrk_rate(energy_ev, nc, nh, channel):
    """I eq. 4 / II eq. 1: k0 (1 - E0/E)^(3N - 7) [s^-1]; zero without the H it needs."""
    k0, e0 = CHANNELS[channel]
    if (channel == "H" and nh < 1) or (channel == "H2" and nh < 2):
        return np.zeros_like(np.asarray(energy_ev, dtype=float))
    return rates.rrk(energy_ev, 3 * (nc + nh) - 6, k0, e0)


def ion_yield(energy_ev, ip_ev):
    """I eq. 11, piecewise linear, size independent (Verstraete+90)."""
    E = np.asarray(energy_ev, dtype=float)
    y = np.select([E < 8.9, E < 11.0], [0.10 * (E - ip_ev) / (8.90 - ip_ev), 0.25 * E - 2.125],
                  0.0596 * E - 0.03)
    return np.where(E >= ip_ev, np.clip(y, 0.0, 1.0), 0.0)


def radius_cm(nc):
    """a = 0.9 N_C^1/2 A (I eq. 13, II eq. 4)."""
    return 0.9e-8 * np.sqrt(nc)


def ionisation_energy(name):
    """I Table 2, else I eq. 12: IP = 4.8 + 10.9 / a[A] eV."""
    return TABLE2_IP.get(name, 4.8 + 10.9 / (radius_cm(SPECIES[name][0]) * 1e8))


def recombination_rate(nc, temperature_k, sticking=0.30, disc=0.8):
    """II eq. 9 with e^2/a: disc P pi a^2 (8kT/pi m_e)^1/2 (1 + e^2/(a kT)) [cm^3 s^-1].

    The printed eq. 9 has 2e^2/a, but the P = 0.3 fit to benzene+ (1e-6 at
    300 K), Fig. 5 and eq. 10's bracket all need e^2/a. Eq. 10's prefactor
    2.13e-6 is 3.3x below eq. 9 (card [electrons].consistency_check).
    """
    kb, me, e = 1.380649e-16, 9.1093837e-28, 4.80320471e-10
    a = radius_cm(nc)
    v = np.sqrt(8.0 * kb * temperature_k / (np.pi * me))
    return disc * sticking * np.pi * a**2 * v * (1.0 + e**2 / (a * kb * temperature_k))


def set1_field():
    """Draine (1978) as I eqs. 18-19, x 4 pi: photons cm^-2 s^-1 eV^-1 on E_GRID.

    Eq. 18 above 5 eV is Draine's per-steradian photon flux. Eq. 19 (2.8-5
    eV) is taken as per steradian too; it falls ~2x below eq. 18 at 5 eV, and
    the card flags this choice.
    """
    E = E_GRID
    eq18 = 6.919e3 * E**3 - 2.152e5 * E**2 + 1.658e6 * E
    eq19 = 4.89085e6 * E**4 - 9.133074e7 * E**3 + 6.36590e8 * E**2 - 1.96811e9 * E + 2.28706e9
    F = 4.0 * np.pi * np.where(E >= 5.0, eq18, eq19)
    return fields.RadiationField(E, np.clip(F, 0.0, None), "Allain+96 Set 1 (Draine 1978)")


SET2_FILE = "pahchem/allain96_set2_field.csv"


def set2_proxy():
    """Stand-in for Set 2 (Gondhalekar+80 + Mathis+83): Set 1 / 1.6. Flagged: not their field."""
    f = set1_field()
    return fields.RadiationField(f.energy_ev, f.photon_flux / SET_RATIO, "Set 1 / 1.6 (Set 2 proxy)")


def set2_field(proxy_ok=True):
    """Set 2, the field of paper II: I Fig. 5 digitised into ``SET2_FILE`` if present.

    The file has columns energy_eV, photon_flux [photons cm^-2 s^-1 eV^-1,
    integrated over 4 pi]. Without it, ``set2_proxy()`` if ``proxy_ok``,
    else FileNotFoundError.
    """
    try:
        t = pd.read_csv(_paths.get_external_data_path(SET2_FILE), comment="#")
    except FileNotFoundError:
        if proxy_ok:
            return set2_proxy()
        raise
    return fields.RadiationField(t["energy_eV"].to_numpy(), t["photon_flux"].to_numpy(),
                                 "Allain+96 Set 2 (I Fig. 5)")


# ---------------------------------------------------------------------------
# Molecular inputs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _malloci():
    return pd.read_csv(_paths.get_external_data_path("pahchem/allain96_sigma_malloci2007.csv"), comment="#")


def sigma(name, charge, nh=None, energy_ev=E_GRID):
    """Absorption cross section [cm^2]: Malloci+07 where bundled, else Li & Draine.

    Dehydrogenated: x (2.5 N_H + 8 N_C)/(2.5 N_H0 + 8 N_C) (II eq. 3).
    """
    nc, nh0, malloci, _ = SPECIES[name]
    E = np.asarray(energy_ev, dtype=float)
    if malloci is not None:
        t = _malloci()
        s = np.interp(E, t["energy_eV"], t[f"{malloci}_{1 if charge > 0 else 0}"]) * 1e-18
    elif name == "C6H6":
        raise ValueError("no gas-phase sigma bundled for benzene (Joblin 1992 not available)")
    else:
        s = photo.sigma_li_draine(nc, charge, E)
    nh = nh0 if nh is None else nh
    return s * (2.5 * nh + 8 * nc) / (2.5 * nh0 + 8 * nc)


def strip_h_modes(species, nh):
    """``species`` with N_H0 - nh C-H stretches, in-plane and out-of-plane bends removed.

    Like ``PAHSpecies.stripped``, but a band with too few modes loses all it
    has and the rest are taken from the next-most-intense C-H window, so
    3N - 6 holds for small PAHs too.
    """
    n = species.nh - nh
    f, A = np.asarray(species.freqs_cm, float), np.asarray(species.einstein_a, float)
    keep = np.ones(f.size, bool)

    def drop(mask, count):
        cand = np.nonzero(keep & mask)[0]
        take = cand[np.argsort(A[cand])[::-1][:count]]
        keep[take] = False
        return count - take.size

    owed = sum(drop((f >= lo) & (f < hi), n) for lo, hi in ((2900.0, 3200.0), (1000.0, 1350.0), (650.0, 1000.0)))
    drop(f >= 650.0, owed)
    return PAHSpecies(species.nc, nh, species.charge, f[keep], A[keep], f"{species.source}, Allain C-H strip to {nh}")


def k_ir(species, energy_ev, multi_quanta=None):
    """IR photon emission rate [s^-1] at internal energy E (I eqs. 8-10, 17).

    sum_i A_i <n_i>(T*), with T* from U(T*) = E. For N_C > 24, sum_i A_i
    (Allain neglect the occupation factor there).
    """
    multi_quanta = species.nc <= MULTI_QUANTA_MAX_NC if multi_quanta is None else multi_quanta
    A = np.asarray(species.einstein_a, float)
    E = np.atleast_1d(np.asarray(energy_ev, float))
    if not multi_quanta:
        return np.full(E.shape, A.sum())
    t = np.logspace(0.5, 4.5, 3000)
    U = dos.thermal_energy(species.freqs_cm, t) / dos.CM_PER_EV
    T = np.interp(E, U, t)
    return np.array([cooling.thermal_mode_rates(species.freqs_cm, A, Ti).sum() for Ti in T])


# ---------------------------------------------------------------------------
# One state: first-event yields and per-molecule rates
# ---------------------------------------------------------------------------

class AllainState:
    """One (species, Z, N_H) state: Allain's yields Y_i(E) on E_GRID."""

    def __init__(self, name, modes, charge=0, nh=None, energy_ev=E_GRID):
        nc, nh0, _, _ = SPECIES[name]
        self.name, self.nc, self.charge = name, nc, charge
        self.nh = nh0 if nh is None else nh
        self.energy_ev = E = np.asarray(energy_ev, float)
        self.k = {c: rrk_rate(E, nc, self.nh, c) for c in CHANNELS}
        sp = modes if self.nh == nh0 else strip_h_modes(modes, self.nh)
        self.k_ir = k_ir(sp, E)
        self.y_ion = ion_yield(E, ionisation_energy(name)) if charge == 0 else np.zeros_like(E)
        k_diss = sum(self.k.values())
        self.yields = {c: (1.0 - self.y_ion) * k / (self.k_ir + k_diss) for c, k in self.k.items()}
        self.sigma = sigma(name, charge, self.nh, E)

    def rates(self, field):
        """Per-molecule photo-rates [s^-1] in ``field``: H, H2, C2H2 loss and ionisation."""
        F = field.flux_at(self.energy_ev)
        out = {c: float(np.trapezoid(self.sigma * F * y, self.energy_ev)) for c, y in self.yields.items()}
        out["ion"] = float(np.trapezoid(self.sigma * F * self.y_ion, self.energy_ev))
        return out


def load_modes(name, states_dir):
    """The neutral PAHdb mode set Allain's ``name`` maps to (module docstring)."""
    fname = SPECIES[name][3]
    if fname is None:
        raise ValueError(f"{name}: no PAHdb mode set")
    return PAHSpecies.from_states_file(states_dir / fname)


# ---------------------------------------------------------------------------
# Paper II: the (Z, N_H) steady state, destruction and survival
# ---------------------------------------------------------------------------

class AllainFamily:
    """All (Z = 0, +1) x (N_H = 0..N_H0) states of one species; photo-rates at chi = 1."""

    def __init__(self, name, modes, field=None):
        self.name = name
        self.nc, self.nh0 = SPECIES[name][:2]
        self.field = set2_field() if field is None else field
        self.photo = {(z, n): AllainState(name, modes, z, n).rates(self.field)
                      for z in (0, 1) for n in range(self.nh0 + 1)}

    def network(self, n_h, n_e, temperature_k, chi):
        """II eqs. 13-14 as a ``PAHNetwork``; returns (network, C-loss rate per state [s^-1])."""
        nh = np.arange(self.nh0 + 1)
        net = PAHNetwork(charges=(0, 1), nh_values=nh)
        get = lambda key: chi * np.array([[self.photo[(z, n)][key] for n in nh] for z in (0, 1)])
        scale = np.sqrt(self.nc / 6.0) if self.nc > 6 else 1.0
        add = np.array([1.7e-11, 1.0e-9])[:, None] * scale * np.ones((2, nh.size))
        T = temperature_k
        abst = (4.15e-10 * np.exp(-8060.0 / T) if T >= 1000.0 else 5e-12 * np.exp(-4080.0 / T))
        abst = abst * scale * (nh / self.nh0)[None, :] * np.array([1.0, 1.0e-9 / 1.7e-11])[:, None]
        n_hplus = max(n_e - 5e-4 * n_h, 0.0)
        below_top = nh < self.nh0
        net.add("photo H loss", get("H"), dnh=-1)
        net.add("photo H2 loss", get("H2"), dnh=-2, h2_released=1)
        net.add("photoionisation", np.vstack([get("ion")[0], np.zeros(nh.size)]), dz=+1)
        net.add("recombination", np.vstack([np.zeros(nh.size),
                                            np.full(nh.size, recombination_rate(self.nc, T) * n_e)]), dz=-1)
        net.add("H addition", np.where(below_top[None, :], add * n_h, 0.0), dnh=+1)
        net.add("H abstraction", abst * n_h, dnh=-1, h2_released=1)
        net.add("H+ addition", np.vstack([np.where(below_top, add[1] * n_hplus, 0.0), np.zeros(nh.size)]),
                dz=+1, dnh=+1)
        net.add("H+ abstraction", np.vstack([abst[1] * n_hplus, np.zeros(nh.size)]), dz=+1, dnh=-1,
                h2_released=1)
        return net, get("C2H2")

    def steady(self, medium):
        """Steady state in one of ``MEDIA``: fractions, alpha_ion, alpha_H and tau_UV [yr]."""
        n_h, n_e, T, chi = MEDIA[medium]
        net, k_c = self.network(n_h, n_e, T, chi)
        f = net.steady_state()
        nh = np.arange(self.nh0 + 1)
        k_uv = float(np.sum(f * k_c))
        return {"f": f, "alpha_ion": float(f[1].sum()),
                "alpha_H": float(f.sum(axis=0) @ nh / self.nh0),
                "k_C_uv": k_uv, "tau_uv_yr": 1.0 / (k_uv * YR),
                "tau_neutral_yr": 1.0 / (k_c[0, -1] * YR), "tau_bare_cation_yr": 1.0 / (k_c[1, 0] * YR)}


def critical_size(nc_values, k_c_values, n_h):
    """Smallest N_C with k_C/(n_H X_C) <= K_C (I eqs. 25-26), interpolating log k linearly in N_C."""
    x = np.log10(np.asarray(k_c_values) / (n_h * X_C)) - np.log10(K_C)
    nc = np.asarray(nc_values, float)
    for i in range(len(nc) - 1):
        if x[i] > 0 >= x[i + 1]:
            return float(np.interp(0.0, [x[i + 1], x[i]], [nc[i + 1], nc[i]]))
    return float(nc[0]) if x[0] <= 0 else np.inf
