"""IR emission of a PAH family in an environment, weighted by its (Z, N_H) populations (M7b).

``FamilyEmission`` couples a ``baseline.BaselineModel`` (whose network gives the
steady-state fractions f(Z, N_H)) to the emission of every state:

- **Modes.** Each state emits with the PAHdb modes of its own charge
  (``charge_species``). Below N_H0 the C-H modes are stripped per missing H
  (``PAHSpecies.stripped``), so the C-H bands weaken. Above N_H0 the parent's
  modes are kept: aliphatic C-H bands (3.4 um) are not modelled.
- **Heating.** Absorption is the baseline's: the Li & Draine sigma of charge Z,
  photons that ionise do not heat. Each state gets its own multiphoton steady
  state P(E) (``energy_me``), and its band photon rates are R_i = sum_j P_j k_i(E_j).
- **The family spectrum** is sum_states f(Z, N_H) h nu_i R_i, the power per
  molecule. Its total equals the absorbed heating power, sum f r_heat, to the
  accuracy of each steady state.

The emission does not feed back on the network: the baseline's own cascades
(parent modes) set f.
"""

from __future__ import annotations

import numpy as np

from pycalima.pahchem import dos, emission, energy_me, photo
from pycalima.pahchem.baseline import E_PH

#: Maragkoudakis+20 Table 1 wavebands [um], also used for the family band ratios
BANDS_UM = {"3.3": (3.1, 3.5), "6.2": (6.1, 6.8), "7.7": (7.0, 8.2), "8.6": (8.2, 8.9),
            "11.2": (10.5, 11.6), "12.7": (12.3, 13.3)}


class FamilyEmission:
    """Population-weighted emission of one parent family.

    Parameters
    ----------
    model : baseline.BaselineModel
    charge_species : dict
        Z -> PAHSpecies of the parent at that charge, for every Z of ``model``.
    de_cm, e_max_ev : float
        Grid of the emission cascades (energy_me.CascadeModel).
    """

    def __init__(self, model, charge_species, de_cm=100.0, e_max_ev=30.0):
        missing = set(model.charges) - set(charge_species)
        if missing:
            raise ValueError(f"no modes for charges {sorted(missing)}")
        self.model, self.charge_species = model, charge_species
        self.de_cm, self.e_max_cm = de_cm, e_max_ev * dos.CM_PER_EV
        self._cascades = {}

    def state_species(self, z, nh):
        sp = self.charge_species[z]
        return sp.stripped(nh) if nh < sp.nh else sp

    def _cascade(self, z, nh):
        key = (z, min(nh, self.charge_species[z].nh))
        if key not in self._cascades:
            sp = self.state_species(*key)
            self._cascades[key] = (sp, energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, self.e_max_cm, self.de_cm))
        return self._cascades[key]

    def state_emission(self, env):
        """{(Z, N_H): (freqs_cm, band photon rates R_i [s^-1], absorbed heating power [cm^-1 s^-1])}."""
        field = env.field()
        out = {}
        for z in self.model.charges:
            sigma = photo.sigma_li_draine(self.model.nc, z, E_PH) * self.model.p.sigma_scale
            for nh in self.model.nh_values:
                sp, cm = self._cascade(z, nh)
                _, h = self.model._absorption(field, sigma, z, nh)
                ss = cm.steady_state(E_PH, h, {})
                out[(z, nh)] = (sp.freqs_cm, emission.steady_state_emission(cm, ss), ss.absorbed_power)
        return out

    def fractions(self, env, photo_rates=None):
        """f(Z, N_H) of the baseline network at ``env``, as a dict."""
        pr = self.model.photo_rates(env) if photo_rates is None else photo_rates
        f = self.model.network(pr, env).steady_state()
        return {(z, nh): float(f[iz, inh]) for iz, z in enumerate(self.model.charges)
                for inh, nh in enumerate(self.model.nh_values)}

    def spectrum(self, env, grid_cm, fractions=None, fwhm_cm=15.0, shift_cm=-15.0):
        """Power per molecule per cm^-1 [cm^-1 s^-1 per cm^-1], and the total absorbed heating power.

        ``fractions`` (Z, N_H) -> f defaults to the network steady state.
        """
        f = self.fractions(env) if fractions is None else fractions
        states = self.state_emission(env)
        s = np.zeros(np.size(grid_cm))
        absorbed = 0.0
        for key, w in f.items():
            if w <= 0.0:
                continue
            freqs, rates, p_abs = states[key]
            s += w * emission.spectrum(freqs, freqs * rates, grid_cm, fwhm_cm, shift_cm)
            absorbed += w * p_abs
        return s, absorbed


def band_fluxes(grid_cm, spectrum_per_cm, bands=BANDS_UM):
    """{band: integral of the spectrum over the waveband} on a uniform wavenumber grid."""
    g = np.asarray(grid_cm)
    d = g[1] - g[0]
    return {b: float(spectrum_per_cm[(g >= 1e4 / hi) & (g <= 1e4 / lo)].sum() * d) for b, (lo, hi) in bands.items()}
