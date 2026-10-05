"""Andrews, Candian & Tielens (2016) photo-rates, emulated with pahchem.

Their dissociation rates (Sect. 2.4.3) are built in three steps:

1. A temperature distribution G_n(T) of the irradiated molecule with no
   dissociation sink: the Bakes et al. (2001) Poisson cooling, iterated over
   photon events until it converges (their eqs. 4-6).
2. Channel averages k_i = int k_i(T) G_n(T) dT (eq. 7), with the
   effective-temperature Arrhenius k(T) and, for IR, the thermal rate
   k_IR = 4 pi sum sigma_i B(nu_i, T)/E (eq. 3). Table B and Fig. 6 are these
   k_i.
3. Branching P_i = k_i / (k_H + k_H2 + k_IR) (eq. 8), and the network rate
   r_abs P_i.

Here the converged G_n is the stationary internal-energy distribution of
``energy_me`` without sinks, which includes multiphoton heating. Two inputs the
paper leaves implicit:

- **Ionising photons heat as well.** G(T, E) is built from all absorbed
  photons. With ionising photons removed from the heating, Table B is missed
  by -0.5 to -1.5 dex; with them included it is reproduced to ~0.5 dex
  (pahchem-spec.md, M5).
- **k_IR in eq. 8.** In their thermal (Purcell) picture a molecule cools
  continuously from T_max(E) down to T_i, "the temperature T_peak at which
  dT/dt = r" (numerically, K s^-1 against s^-1). Along that path
  int k_IR G dT = int h(e) ln(e / U(T_i)) de, because k_IR = (dE/dt)/E.
  That is the k_IR used here. Averaging the thermal k_IR over the discrete
  stationary P(E) would not be equivalent: there the molecule stops below its
  lowest mode, where the thermal k_IR is not small, and every P_i drops to
  ~1e-4 (coronene, G0 = 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from pycalima.pahchem import cooling, dos, energy_me, photo, rates

# Table 2: (E_act [eV], dS [J K^-1 mol^-1])
H_LOSS_EVEN, H_LOSS_ODD, H2_LOSS = (4.6, 44.8), (4.1, 55.6), (3.52, -53.1)
H_LOSS_SUPERH = {"Z<=0": (1.4, 55.6), "Z>0": (1.55, 55.6)}


def channels(nh, nh0, n_solo, charge):
    """Andrews+16 Table 2 channels of state (Z, N_H): {name: (E_act, dS)}."""
    if nh > nh0:
        return {"H": H_LOSS_SUPERH["Z>0" if charge > 0 else "Z<=0"]}   # H2 by Eley-Rideal instead
    out = {}
    if nh > 0:
        out["H"] = H_LOSS_EVEN if nh % 2 == 0 else H_LOSS_ODD
    if nh % 2 == 0 and nh > n_solo and nh >= 2:                     # H2 from a duo ring
        out["H2"] = H2_LOSS
    return out


@dataclass
class Andrews16Rates:
    r_abs: float          # all absorptions [s^-1]
    k_eq7: dict           # channel -> eq. 7 average (Table B quantity) [s^-1]
    k_ir_eq7: float       # eq. 7 average of k_IR along the cooling path to T_i [s^-1]
    t_i: float            # low-T cut of G [K]
    branching: dict       # channel -> P_i (eq. 8)

    @property
    def network_rates(self) -> dict:
        """r_abs P_i [s^-1]: what their network uses."""
        return {c: self.r_abs * p for c, p in self.branching.items()}


class Andrews16State:
    """One (Z, N_H) state with a mode set, on an ``energy_me`` grid."""

    def __init__(self, species, nh0, n_solo, charge, e_max_ev=50.0, de_cm=100.0):
        self.species, self.charge, self.nh = species, charge, species.nh
        self.cm = energy_me.CascadeModel(species.freqs_cm, species.einstein_a,
                                         e_max_cm=e_max_ev * dos.CM_PER_EV, de_cm=de_cm)
        E = self.cm.energy_cm
        self.k = {c: rates.arrhenius_effective_temperature(species.freqs_cm, E, Ea * dos.CM_PER_EV, dS)
                  for c, (Ea, dS) in channels(species.nh, nh0, n_solo, charge).items()}
        # canonical T(E) by inverting U(T) on a log-T grid
        t_grid = np.logspace(0.5, 4.5, 2000)
        U = dos.thermal_energy(species.freqs_cm, t_grid)
        T = np.where(E > 0, np.interp(E, U, t_grid), 0.0)
        A = np.asarray(species.einstein_a)
        x = species.freqs_cm[:, None] / (dos.KB_CM * np.maximum(T, 1e-3)[None, :])
        power = (species.freqs_cm * A) @ (1.0 / np.expm1(np.minimum(x, 700.0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.k_ir_thermal = np.where(E > 0, power / E, 0.0)      # eq. 3 [s^-1]
        self._T = T

    def _cooling_rate_k_per_s(self, T):
        f, A = self.species.freqs_cm, self.species.einstein_a
        power = cooling.cooling_rate(f, cooling.thermal_mode_rates(f, A, T))
        dT = 1e-3 * T
        cv = (dos.thermal_energy(f, T + dT) - dos.thermal_energy(f, T - dT)) / (2 * dT)
        return power / cv

    def t_peak(self, r_abs):
        """T_i: where the thermal cooling rate dT/dt [K s^-1] equals r_abs [s^-1]."""
        f = lambda t: np.log(self._cooling_rate_k_per_s(t)) - np.log(r_abs)
        lo = 2.0
        while f(lo) > 0 and lo > 0.05:
            lo /= 2.0
        return brentq(f, lo, 3000.0) if f(lo) < 0 else lo

    def rates(self, photon_energy_ev, sigma_cm2, field) -> Andrews16Rates:
        E_ph = np.asarray(photon_energy_ev, dtype=float)
        h = photo.heating_spectrum(field, sigma_cm2, np.zeros_like(E_ph), E_ph)   # all absorptions heat
        r_abs = float(np.trapezoid(h, E_ph))
        P = self.cm.steady_state(E_ph, h, {}).probability
        k7 = {c: float(P @ k) for c, k in self.k.items()}
        t_i = self.t_peak(r_abs)
        e_i = dos.thermal_energy(self.species.freqs_cm, t_i) / dos.CM_PER_EV       # eV
        k_ir7 = float(np.trapezoid(h * np.log(E_ph / e_i), E_ph))
        total = sum(k7.values()) + k_ir7
        return Andrews16Rates(r_abs, k7, k_ir7, t_i, {c: v / total for c, v in k7.items()})


# ---------------------------------------------------------------------------
# The (Z, N_H) network of one parent (Appendix A, Sects. 2.3-2.4)
# ---------------------------------------------------------------------------

#: Table A.1 at N_H0: ionisation energies of Z -> Z+1 (Z = -1 is the EA).
IONISATION = {"C24": {-1: 0.47, 0: 7.20, 1: 11.50},
              "C54": {-1: 1.44, 0: 6.14, 1: 8.91, 2: 12.94},
              "C96": {-1: 0.56, 0: 5.68, 1: 8.24, 2: 10.80, 3: 13.36}}
PARENTS = {"C24": (24, 12, 0), "C54": (54, 18, 6), "C96": (96, 24, 12)}   # N_C, N_H0, N_solo
DELTA_IP = 0.1          # eV per missing H (eq. 1)
CORONENE_GEOMETRIC_A2 = 35.7


class Andrews16Model:
    """Andrews+16 network for one parent, with the photo-rates of ``Andrews16State``.

    Parameters
    ----------
    parent : {"C24", "C54", "C96"}
    mode_files : dict
        (Z, N_H) -> PAHSpecies for every available mode set. Each state uses
        the closest N_H of the same charge, or of any charge if there is none
        (their rule, Sect. 2.3.3).
    """

    def __init__(self, parent, mode_files, e_max_ev=50.0, de_cm=100.0, temperature_template=15000):
        from pycalima.pahchem import photo as _photo

        self.parent = parent
        self.nc, self.nh0, self.n_solo = PARENTS[parent]
        self.charges = tuple(sorted(IONISATION[parent])) + (max(IONISATION[parent]) + 1,)
        self.nh_values = tuple(range(0, self.nh0 + 5))
        self._modes = mode_files
        self._e_max, self._de = e_max_ev, de_cm
        self._states = {}
        self._E_ph = np.linspace(5.0, 13.6, 345)
        self._sigma = {z: _photo.sigma_li_draine(self.nc, z, self._E_ph) for z in self.charges}
        self._teff = temperature_template

    def ionisation_energy(self, z, nh):
        """Energy to go from charge z to z + 1 at N_H (eq. 1); None if not in Table A.1."""
        ip = IONISATION[self.parent].get(z)
        return None if ip is None else ip + DELTA_IP * (self.nh0 - nh)

    def _species_for(self, z, nh):
        same = [k for k in self._modes if k[0] == z] or list(self._modes)
        key = min(same, key=lambda k: (abs(k[1] - nh), abs(k[0] - z)))
        return key

    def _state(self, z, nh):
        key = self._species_for(z, nh)
        if key not in self._states:
            self._states[key] = Andrews16State(self._modes[key], self.nh0, self.n_solo, key[0],
                                               self._e_max, self._de)
        st = self._states[key]
        E = st.cm.energy_cm
        k = {c: rates.arrhenius_effective_temperature(st.species.freqs_cm, E, Ea * dos.CM_PER_EV, dS)
             for c, (Ea, dS) in channels(nh, self.nh0, self.n_solo, z).items()}
        return st, k

    def photo_rates(self, g0):
        """{(Z, N_H): {"H": r P_H, "H2": r P_H2, "ion": k_ion}} at Habing ``g0``."""
        from pycalima.pahchem import fields as _fields, photo as _photo

        field = _fields.kurucz(self._teff, g0)
        E_ph = self._E_ph
        P_cache = {}
        out = {}
        for z in self.charges:
            sigma = self._sigma[z]
            h = _photo.heating_spectrum(field, sigma, np.zeros_like(E_ph), E_ph)
            r_abs = float(np.trapezoid(h, E_ph))
            for nh in self.nh_values:
                st, k = self._state(z, nh)
                ckey = (id(st), z)
                if ckey not in P_cache:
                    P_cache[ckey] = (st.cm.steady_state(E_ph, h, {}).probability, st.t_peak(r_abs))
                P, t_i = P_cache[ckey]
                k7 = {c: float(P @ kk) for c, kk in k.items()}
                e_i = dos.thermal_energy(st.species.freqs_cm, t_i) / dos.CM_PER_EV
                k_ir7 = float(np.trapezoid(h * np.log(E_ph / e_i), E_ph))
                total = sum(k7.values()) + k_ir7
                rec = {c: r_abs * v / total for c, v in k7.items()}
                ip = self.ionisation_energy(z, nh)
                rec["ion"] = (0.0 if ip is None or ip >= 13.6
                              else _photo.ionisation_rate(field, sigma, _photo.yield_jochims1996(E_ph, ip), E_ph))
                out[(z, nh)] = rec
        return out

    def network(self, photo_rates, n_h, n_e, temperature_k):
        from pycalima.pahchem import chemistry
        from pycalima.pahchem.network import PAHNetwork

        net = PAHNetwork(self.charges, self.nh_values)
        Z = np.array(self.charges)[:, None] * np.ones((1, len(self.nh_values)), int)
        NH = np.array(self.nh_values)[None, :] * np.ones((len(self.charges), 1), int)
        get = lambda key: np.array([[photo_rates[(z, n)].get(key, 0.0) for n in self.nh_values]
                                    for z in self.charges])
        top_z, top_nh = max(self.charges), max(self.nh_values)
        net.add("H loss", get("H"), dnh=-1)
        net.add("H2 loss", get("H2"), dnh=-2, h2_released=1)
        net.add("photoionisation", np.where(Z < top_z, get("ion"), 0.0), dz=+1)
        capture = np.array([[chemistry.electron_capture(self.nc, z, temperature_k) for _ in self.nh_values]
                            for z in self.charges])
        net.add("recombination", np.where(Z > min(self.charges), capture * n_e, 0.0), dz=-1)
        # H addition (Sect. 2.4.4)
        sigma_geo = CORONENE_GEOMETRIC_A2 * self.nc / 24.0
        barrier = np.select([NH == self.nh0, NH == self.nh0 + 2], [0.06, 0.035], 0.0)
        neutral = np.where(NH < self.nh0,
                           0.07 * chemistry.cross_section_rate(temperature_k, sigma_geo),
                           chemistry.cross_section_rate(temperature_k, sigma_geo, barrier))
        can_add = NH < top_nh
        net.add("H addition, cations", np.where((Z > 0) & can_add, 1.4e-10 * n_h, 0.0), dnh=+1)
        net.add("H addition, neutrals", np.where((Z == 0) & can_add, neutral * n_h, 0.0), dnh=+1)
        net.add("associative detachment", np.where((Z == -1) & can_add, 7.8e-10 * n_h, 0.0), dz=+1, dnh=+1)
        er = chemistry.cross_section_rate(temperature_k, 0.06) * n_h
        net.add("Eley-Rideal", np.where(NH > self.nh0, er, 0.0), dnh=-1, h2_released=1)
        return net
