"""Montillaud, Joblin & Toublanc (2013) run through the pahchem engine.

The inputs come from ``method_cards/montillaud13.toml``; the card's
``[run].deviations`` lists where pahchem cannot use exactly what the paper
used. The physics is pahchem's own:

- an inverse-Laplace k_H(E) = A rho(E - E0)/rho(E), with E0 = 4.8 / 3.2 eV for
  even / odd N_H; the N_H0 + 1 state dissociates like N_H0 - 1;
- the microcanonical IR cascade and multiphoton steady state of
  ``energy_me``;
- Le Page et al. (2001) ionisation yields;
- charges 0 and +1 only;
- cation H addition at the Table 4 rates, none for neutrals (standard model).

``modes`` chooses the mode set of each hydrogenation state, which the paper
does not state:
- "parent": the parent's modes for every state;
- "stripped": ``PAHSpecies.stripped``.
The choice moves the even-N_H rates by ~100x (pahchem-spec.md, M5).

``hydrogenation_grid`` computes the quantity of their Fig. 10: the fraction of
a parent that is totally dehydrogenated, on a (G0, nH) grid at their
conditions.
"""

from __future__ import annotations

import numpy as np

from pycalima.pahchem import dos, energy_me, fields, photo, rates
from pycalima.pahchem.cards import load_card
from pycalima.pahchem.network import PAHNetwork

E_PH = np.linspace(5.0, 13.6, 345)
PARENTS = {"C24": (24, 12), "C54": (54, 18), "C66": (66, 20), "C96": (96, 24)}


class Montillaud13:
    """One parent (e.g. ``"C54"``) under the Montillaud+13 card.

    Parameters
    ----------
    species : PAHSpecies
        Modes of the normally hydrogenated parent.
    parent : str
        Key into the card tables: "C24", "C54", "C66" or "C96".
    modes : {"parent", "stripped"}
        Mode set of the dehydrogenated states.
    e_max_ev, de_cm : float
        Energy grid of the master equation.
    """

    def __init__(self, species, parent, modes="parent", e_max_ev=45.0, de_cm=100.0):
        if modes not in ("parent", "stripped"):
            raise ValueError("modes must be 'parent' or 'stripped'")
        self.card = load_card("montillaud13")
        self.parent = parent
        self.nc, self.nh0 = PARENTS[parent]
        self.modes = modes
        d = self.card["dissociation"]
        self._A = d["A_diss"]["value"]
        self._E0 = {"even": d["E0_even"]["value"], "odd": d["E0_odd"]["value"]}
        self._e_max, self._de = e_max_ev * dos.CM_PER_EV, de_cm
        self._states = {}
        self._parent_species = species
        self.ip = self.card["ionisation"]["ip"][parent]
        self.sigma = {z: photo.sigma_li_draine(self.nc, z, E_PH) for z in (0, 1)}

    def _state(self, nh):
        """(CascadeModel, k_H on its grid) for a state with nh > 0 hydrogens."""
        key = nh if self.modes == "stripped" else "parent"
        parity = "even" if nh % 2 == 0 else "odd"
        if (key, parity) not in self._states:
            sp = self._parent_species if self.modes == "parent" else self._parent_species.stripped(nh)
            cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=self._e_max, de_cm=self._de)
            E, counts = dos.beyer_swinehart(sp.freqs_cm, cm.energy_cm[-1] + 200.0)
            k = rates.inverse_laplace_arrhenius(E, counts, self._A, self._E0[parity] * dos.CM_PER_EV)
            self._states[(key, parity)] = (cm, k[np.rint(cm.energy_cm).astype(int)])
        return self._states[(key, parity)]

    def photo_rates(self, g0):
        """(k_H[Z, N_H], k_ion[Z]) [s^-1] in the Mathis ISRF at ``g0``."""
        field = fields.mathis1983(g0=g0)
        y = {0: photo.yield_lepage2001(E_PH, self.ip), 1: np.zeros_like(E_PH)}   # no Z = 2
        k_h = np.zeros((2, self.nh0 + 2))
        k_ion = np.zeros(2)
        for z in (0, 1):
            h = photo.heating_spectrum(field, self.sigma[z], y[z], E_PH)
            k_ion[z] = photo.ionisation_rate(field, self.sigma[z], y[z], E_PH)
            cache = {}
            for nh in range(1, self.nh0 + 1):
                cm, k = self._state(nh)
                if (id(cm), id(k)) not in cache:
                    cache[(id(cm), id(k))] = cm.steady_state(E_PH, h, {"H": k}).rates["H"]
                k_h[z, nh] = cache[(id(cm), id(k))]
            k_h[z, self.nh0 + 1] = k_h[z, self.nh0 - 1]          # k(NH0+1) = k(NH0-1), card
        return k_h, k_ion

    def network(self, photo_rates, n_h_cm3, n_e_cm3, temperature_k):
        run = self.card["run"]
        k_h, k_ion = photo_rates
        nh = np.arange(self.nh0 + 2)
        net = PAHNetwork(charges=(0, 1), nh_values=nh)
        alpha = run["recombination_alpha_300K"][self.parent] * np.sqrt(300.0 / temperature_k)
        add_cat = np.where(nh % 2 == 0, run["cation_h_addition_even"][self.parent],
                           run["cation_h_addition_odd"]["value"])
        add_cat[self.nh0 + 1] = run["cation_h_addition_superh"]["value"]
        add = np.array([np.full(nh.size, run["neutral_h_addition_standard"]["value"]), add_cat]) * n_h_cm3
        net.add("H loss", k_h, dnh=-1)
        net.add("photoionisation", np.array([np.full(nh.size, k_ion[0]), np.zeros(nh.size)]), dz=+1)
        net.add("recombination", np.array([np.zeros(nh.size), np.full(nh.size, alpha * n_e_cm3)]), dz=-1)
        net.add("H addition", add, dnh=+1)
        return net

    def hydrogenation_grid(self, g0_values, nh_values_cm3, temperature_k=100.0):
        """Fraction totally dehydrogenated, shape (len(g0), len(nH)), at the Fig. 10 conditions."""
        out = np.zeros((len(g0_values), len(nh_values_cm3)))
        for i, g0 in enumerate(g0_values):
            pr = self.photo_rates(g0)
            for j, n in enumerate(nh_values_cm3):
                f = self.network(pr, 0.5 * n, 1.4e-4 * n, temperature_k).steady_state()
                out[i, j] = f[:, 0].sum()
        return out
