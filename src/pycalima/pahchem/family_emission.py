"""IR emission of a PAH family in an environment, weighted by its (Z, N_H) populations (M7b).

``FamilyEmission`` couples a ``baseline.BaselineModel`` (whose network gives the
steady-state fractions f(Z, N_H)) to the emission of every state:

- **Modes.** Each state emits with the PAHdb modes of its own charge
  (``charge_species``). Below N_H0 the C-H modes are stripped per missing H
  (``PAHSpecies.stripped``), so the C-H bands weaken.
- **Super-hydrogenated states** (N_H > N_H0) emit with the DFT modes of the
  genuine super-H isomer (the parent skeleton with one sp3 CH2 per extra H;
  ``superh_species``, from ``load_superh_species``). These carry the aliphatic
  C-H stretches (3.4 um), the CH2 scissoring (6.9 um) and wag/twist modes, and
  the weakened aromatic C-H bands. PAHdb has only some (Z, N_H): a missing
  state takes the nearest N_H of the same charge, else of the nearest charge
  (``superh_choice`` records which). Without ``superh_species`` the parent's
  modes are used and no aliphatic band appears.
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
from pycalima.pahchem.baseline import E_PH, scaled_einstein_a

#: Wavebands [um]. "3.3" ... "11.2" are Maragkoudakis+20 Table 1 ("3.3" holds both C-H
#: stretches); "3.29" (aromatic) and "3.4" (aliphatic) split it, and "6.9" holds the CH2
#: scissoring, between their 6.2 and 7.7 windows. Positions after the -15 cm^-1 shift.
BANDS_UM = {"3.3": (3.1, 3.5), "3.29": (3.20, 3.36), "3.4": (3.36, 3.60), "6.2": (6.1, 6.8),
            "6.9": (6.8, 7.0), "7.7": (7.0, 8.2), "8.6": (8.2, 8.9), "11.2": (10.5, 11.6),
            "12.7": (12.3, 13.3)}


def load_superh_species(directory, formula):
    """{(Z, N_H): PAHSpecies} of the super-H files ``<formula skeleton>H<n>_<Z>.dat`` in ``directory``.

    ``formula`` is the parent, e.g. "C54H18"; files with more H on the same
    carbon count are returned (model_data/PAH_states/superh, see its README).
    """
    from pathlib import Path

    from pycalima.pahchem.molecule import PAHSpecies

    nc = formula.split("H")[0]
    out = {}
    for p in sorted(Path(directory).glob(f"{nc}H*_*.dat")):
        sp = PAHSpecies.from_states_file(p)
        out[(sp.charge, sp.nh)] = sp
    return out


class FamilyEmission:
    """Population-weighted emission of one parent family.

    Parameters
    ----------
    model : baseline.BaselineModel
    charge_species : dict
        Z -> PAHSpecies of the parent at that charge, for every Z of ``model``.
    superh_species : dict, optional
        (Z, N_H) -> PAHSpecies of genuine super-H isomers (``load_superh_species``).
    de_cm, e_max_ev : float
        Grid of the emission cascades (energy_me.CascadeModel).
    """

    def __init__(self, model, charge_species, superh_species=None, de_cm=100.0, e_max_ev=30.0):
        missing = set(model.charges) - set(charge_species)
        if missing:
            raise ValueError(f"no modes for charges {sorted(missing)}")
        self.model, self.charge_species = model, charge_species
        self.superh = dict(superh_species or {})
        self.de_cm, self.e_max_cm = de_cm, e_max_ev * dos.CM_PER_EV
        self._cascades = {}

    def superh_choice(self, z, nh):
        """The (Z, N_H) of the super-H modes used for state (z, nh > N_H0), or None."""
        if not self.superh:
            return None
        if (z, nh) in self.superh:
            return (z, nh)
        return min(self.superh, key=lambda k: (abs(k[0] - z), abs(k[1] - nh), k[1]))

    def _mode_key(self, z, nh):
        parent = self.charge_species[z]
        if nh < parent.nh:
            return ("stripped", z, nh)
        if nh > parent.nh and self.superh:
            return ("superh",) + self.superh_choice(z, nh)
        return ("parent", z, parent.nh)

    def state_species(self, z, nh):
        kind, zz, n = self._mode_key(z, nh)
        if kind == "superh":
            return self.superh[(zz, n)]
        sp = self.charge_species[zz]
        return sp.stripped(n) if kind == "stripped" else sp

    def _cascade(self, z, nh):
        levels = self.model.rf_levels(z)
        key = self._mode_key(z, nh) + (levels,)
        if key not in self._cascades:
            sp = self.state_species(z, nh)
            # the model's CH-stretch scale, as the population fits use (pahchem-audit C7)
            self._cascades[key] = (sp, energy_me.CascadeModel(sp.freqs_cm, scaled_einstein_a(sp, self.model.p),
                                                              self.e_max_cm,
                                                              self.de_cm, electronic=levels))
        return self._cascades[key]

    def state_emission(self, env):
        """{(Z, N_H): (freqs_cm, IR band photon rates R_i [s^-1], absorbed heating power,
        power carried off by recurrent fluorescence [cm^-1 s^-1])}.

        IR plus RF power equals the absorbed power; the RF levels are the
        baseline's (``BaselineModel.rf_levels``).
        """
        field = env.field()
        out = {}
        for z in self.model.charges:
            sigma = photo.sigma_li_draine(self.model.nc, z, E_PH) * self.model.p.sigma_scale
            for nh in self.model.nh_values:
                sp, cm = self._cascade(z, nh)
                _, h = self.model._absorption(field, sigma, z, nh)
                ss = cm.steady_state(E_PH, h, {})
                rates = emission.steady_state_emission(cm, ss)
                rf_power = float(cm.freqs_cm[cm.n_ir:] @ rates[cm.n_ir:])
                out[(z, nh)] = (sp.freqs_cm, rates[:cm.n_ir], ss.absorbed_power, rf_power)
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
            freqs, rates, p_abs, _ = states[key]
            s += w * emission.spectrum(freqs, freqs * rates, grid_cm, fwhm_cm, shift_cm)
            absorbed += w * p_abs
        return s, absorbed


def band_fluxes(grid_cm, spectrum_per_cm, bands=BANDS_UM):
    """{band: integral of the spectrum over the waveband} on a uniform wavenumber grid."""
    g = np.asarray(grid_cm)
    d = g[1] - g[0]
    return {b: float(spectrum_per_cm[(g >= 1e4 / hi) & (g <= 1e4 / lo)].sum() * d) for b, (lo, hi) in bands.items()}
