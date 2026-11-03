"""The pahchem baseline model of one PAH family, for the sensitivity analysis.

Unlike ``montillaud13`` and ``andrews16``, this is pahchem's own physics, with
each input taken from the laboratory-gated pieces where they exist. The values
and their sources are in ``method_cards/pahchem_baseline.toml``.

- **Dissociation.** RRKM with transition states built from the West et al.
  (2018) iPEPICO (E0, dS(1000 K)) by their convention (5 lowest modes
  scaled). The path degeneracy is N_H for aromatic H loss and 1 for the
  odd-H radical, as their ESI k(E) implies (pahchem-spec.md, M2).
  Superhydrogenated H loss uses E0 = 1.4 / 1.55 eV (Andrews+16).
- **Cascade.** The microcanonical IR cascade and multiphoton steady state of
  ``energy_me``, **with** dissociation sinks: rates are physical, at most
  r_abs.
- **Absorption.** Li & Draine sigma per charge; Jochims+96 yields, except
  cation -> dication with Wenzel+20 where it applies (N_C >= 32); ionising
  photons deposit nothing.
- **Charge and H chemistry.** Table A.1 IP/EA of Andrews+16 with 0.1 eV per
  missing H; BT94 electron capture with the population's sticking
  (``BaselineModel.capture_coefficients``: the cation factor of
  ``Params.capture_model``, and on neutrals the attachment fraction times the
  statistical s_e of ``attachment``); the Andrews+16 H addition,
  associative detachment and Eley-Rideal.

``Params`` holds every knob. ``sensitivity`` perturbs them one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from pycalima.pahchem import chemistry, dos, energy_me, fields, photo, rates
from pycalima.pahchem.andrews16 import CORONENE_GEOMETRIC_A2, DELTA_IP, IONISATION, PARENTS
from pycalima.pahchem.network import PAHNetwork

E_PH = np.linspace(5.0, 13.6, 345)


#: CH stretch modes scaled by ``Params.stretch_scale`` [cm^-1]. The aliphatic CH2/CH3 stretches (2700-3000) are
#: included: the scale is measured on aromatic stretches only, and is assumed to hold for them as well
CH_STRETCH_CM = (2600.0, 3200.0)
#: lab / DFT ratio of the CH stretch to CH oop intensity: -0.29 dex, the median of 35 neutrals
#: (check_lab_band_strengths.py: PAHdb v4.00 6-31G* against PAHdb experimental v3.10, matrix isolation)
LAB_STRETCH_SCALE = 10.0 ** -0.29


def scaled_einstein_a(sp, params):
    """``sp.einstein_a`` with the CH-stretch modes (``CH_STRETCH_CM``) times ``params.stretch_scale``."""
    if params.stretch_scale == 1.0:
        return sp.einstein_a
    f = sp.freqs_cm
    return sp.einstein_a * np.where((f >= CH_STRETCH_CM[0]) & (f <= CH_STRETCH_CM[1]), params.stretch_scale, 1.0)


@dataclass(frozen=True)
class Params:
    # dissociation: (E0 [eV], dS(1000 K) [J/K/mol]); West+18 coronene+, mean 2nd-H, perylene+ H2
    h_even: tuple = (4.41, 12.0)
    h_odd: tuple = (3.16, 10.0)
    h2: tuple = (4.58, 40.0)
    h_superh_e0: tuple = (1.40, 1.55)        # Z <= 0, Z > 0
    h_superh_ds: float = 12.0
    rate_form: str = "rrkm"                  # or "arrhenius_te" (Andrews+16 Table 2)
    modes: str = "parent"                    # or "stripped"
    # photophysics
    sigma_scale: float = 1.0
    # UV absorption (``photo.sigma_uv``): our Octopus TD-DFT library (``uvlib``; the default since 2026-11-01, with the neutrals' bands
    # shifted up 0.4 eV, ad hoc and labelled in docs/physics/pahchem-uv-absorption.md section 5; "octopus_library_tdlda" keeps TD-LDA's
    # positions), Berne+22's Malloci+07 average per C at 5-30 eV with Li & Draine outside ("malloci_average", the default from 2026-10-26),
    # or "li_draine" everywhere (before)
    uv_cross_section: str = "octopus_library"
    ir_scale: float = 1.0                    # proxy for recurrent fluorescence / anharmonic cooling
    yield_model: str = "jochims"             # or "lepage"
    yield_cation: str = "wenzel"             # Z = +1 where Wenzel+20 applies (N_C >= 32, IP2 < 11.3); or "jochims"
    # population only, Z >= +2 (no yield measured): "none" (the lab laws, unity above max(20 eV, IP)), "window" (opt-in
    # bound: capped by photo.yield_coulomb_window, the Coulomb-bound electrons autoionising against relaxation, u_c from
    # the Z = +1 data) or "y2" (WD01's escape fraction, none of them autoionising; excluded at Z = +1)
    coulomb_escape: str = "none"
    # population only: "knockout" (opt-in) removes two electrons in photo.double_fraction of the photoionisations of
    # Z >= 0 above IP(Z) + IP(Z+1) (dz = +2; from the state below the cold limit, a Coulomb explosion); "none" one each
    double_ionisation: str = "none"
    # population only: C 1s absorption (population.kshell_rates): "auger" (the default since 2026-11-03: the photoelectron
    # plus Auger decay, +1 to +3 per absorption, to 10 keV) or "single" (before: one electron each, as the valence, to 500 eV)
    kshell: str = "auger"
    ionising_photons_heat: bool = False
    # charge
    capture_scale: float = 1.0
    detachment: str = "valence"              # anion photodetachment: "valence" (statistical + valence photodetachment; default since 2026-10-25, the only one passing V2 and V14), "statistical" (population: WD01 prompt + detailed-balance delayed), "jochims" (Jochims at the EA, every electron taking E - EA; the default before) or "unity" above it
    # cation capture (population and BaselineModel): "lab_cold" (chemistry.lab_sticking on the photo-ion data, the
    # default since 2026-11-02; measured at N_C 10-16 only, its rise to 1 by N_C ~ 40 an extrapolation), "lab_pahs" (every PAH
    # point), "bt94" (sticking 1, the default before), "lab_trend" (chemistry.lab_recombination_factor) or "wd01" (comparison:
    # population.wd01_capture_ratio). docs/physics/pahchem-gas-coupling.md section 2.2b
    capture_model: str = "lab_cold"
    capture_cap: float | None = None         # upper bound on the lab_trend factor (None: the power law as fitted)
    # charge ladder (population only): "cold_limit" (the default since 2026-11-02: up to floor(0.2 N_C),
    # impact.coulomb_limit; states above the highest computed charge are stand-ins, population.extend_ladder) or
    # "library" (the charges the library computed, its top state holding the higher charges; before)
    charge_ladder: str = "cold_limit"
    capture_t_power: float = 0.0             # cation capture x (T/300 K)^-capture_t_power: 0.25 steepens BT94's ~T^-1/2 to the T^-0.75 of small aromatics in storage rings (Hamberg+11)
    stretch_scale: float = 1.0               # Einstein A of the CH stretches (CH_STRETCH_CM), every cascade; LAB_STRETCH_SCALE for the lab ratio
    recombination_heating: bool = True       # population only: each recombination into a state deposits its threshold
    sticking_neutral: float | str = "statistical"   # or a fixed s_e
    # PAH- + H -> PAH(0, N_H+1) + e (Demarais+12, measured on dehydrogenated anions): "dehydrogenated" (N_H < N_H0
    # only; none in the population's charge chain, which holds N_H0) or "all" (every anion: Andrews+16; before 2026-10-26)
    associative_detachment: str = "dehydrogenated"
    # f with attachment_model "constant": share of classical captures on a neutral that form the anion
    # (``attachment.Attachment.capture_fraction``; autodetachment scales with it by detailed balance). 1 is the classical limit
    attachment_fraction: float = 1.0
    # f per molecule (population.attachment_fraction; the population and BaselineModel): "size" (the
    # default since 2026-11-02: attachment_f_low + (1 - attachment_f_low) / (1 + (attachment_nc_half / N_C)^attachment_nc_power)),
    # "constant" (f = attachment_fraction; 1 before) or "ea" (a logistic in the electron affinity: attachment_f_low + (1 -
    # attachment_f_low) / (1 + exp(-(EA - attachment_ea_half) / attachment_ea_width))). Anchors: ~0.003 for anthracene, perylene
    # and coronene (lab, Carelli+13 upper bounds), ~0.5-0.7 for C60 (Viggiano+10, 400-625 K, a factor 3). Between C24 and C60
    # nothing is measured: the logistic's midpoint and steepness there are an extrapolation, constrained only by the
    # PDRs4All fronts and Wolfire+08 for N_C >~ 50 (docs/physics/pahchem-gas-coupling.md section 2.2b)
    attachment_model: str = "size"
    attachment_ea_half: float = 2.0
    attachment_ea_width: float = 0.2
    attachment_nc_half: float = 50.0
    attachment_nc_power: float = 10.0
    attachment_f_low: float = 0.003
    # ion-PAH collisions (population; Environment.ions): efficiencies on DS87 (anions) and Langevin (neutrals)
    mn_efficiency: float = 0.58
    ion_neutral_efficiency: float = 0.9
    metal_adducts: bool = False              # [M-PAH]+ of Na+, Mg+, Fe+, Si+ below the charge-transfer threshold
    # hydrogen chemistry
    addition_cation: float = 1.4e-10         # cm^3 s^-1
    addition_neutral_scale: float = 1.0
    eley_rideal_a2: float = 0.06
    # carbon loss of the bare skeleton (a key of CARBON_LOSS)
    carbon_loss: str = "leger89"
    # recurrent fluorescence of the charges in RF_LEVELS (H-bearing states; not the bare skeleton)
    recurrent_fluorescence: bool = True
    # numerics
    de_cm: float = 200.0
    e_max_ev: float = 45.0


@dataclass(frozen=True)
class Environment:
    name: str
    g0: float
    n_h: float           # atomic H [cm^-3]
    n_e: float           # [cm^-3]
    temperature_k: float
    teff_k: int | None = 15000   # Kurucz template; None for the Draine (1978) ISRF shape
    field_e_max_ev: float = fields.LYMAN_EV   # Kurucz cut: 13.6 eV for neutral gas; higher to let EUV in (population)
    ions: tuple = ()                          # ((name, n [cm^-3]), ...) of chemistry.IONS, inputs like n_e (population)
    field_e_min_ev: float = 1.0               # lowest photon energy the field reaches (population; 5.0: as before 2026-10-24)
    optical_isrf: float = 1.0                 # old-star light below 5 eV, x the solar circle's, not scaled by g0 (Draine shape, population)
    field_powerlaw: float | None = None       # F_nu ~ nu^-alpha from field_e_min_ev to field_e_max_ev instead of a star (population);
    #                                           field_e_min_ev > 13.6: X-rays only, and g0 is their energy flux in Habing units

    def field(self):
        if self.field_powerlaw is not None:
            return fields.powerlaw(self.field_powerlaw, self.g0, e_min=self.field_e_min_ev, e_max=self.field_e_max_ev)
        return (fields.kurucz(self.teff_k, self.g0, e_max=self.field_e_max_ev) if self.teff_k is not None
                else fields.draine1978(g0=self.g0))


#: Recurrent fluorescence: parent -> {Z: Malloci+07 TD-DFT cross-section file of that cation}.
#: The levels are its bands below 3.1 eV, one per band at the peak, with the integrated f
#: (photo.rf_levels_from_cross_section): C24H12+ (1.70 eV, 0.068), (2.10, 0.008), (2.85, 0.068);
#: C54H18+ (1.05, 0.076), (2.40, 1.08). Equilibrium-geometry values, which Subramani+25 find
#: underestimate measured RF: a lower bound. No spectrum exists for C96H24+ or the dications
#: (pahchem-data-requests, T3), so RF is off there.
RF_SPECTRA = {"C24": {1: "coronene"}, "C54": {1: "circumcoronene"}}      # malloci2007-spectra names

#: Carbon loss from the bare skeleton: name -> (rate form, {fragment: (n_C, A [s^-1] or dS [J/K/mol], E0 [eV])}).
#:
#: - "leger89": Leger+89 Table 2, from graphite sublimation, inverse-Laplace form (grade C; the default).
#: - "c60_cage": C2 loss of C60+ (Matt+01): the slow bound if bare flakes fold into cages first.
#: - "micelotta10b": C2H2 loss from small cations raised to a 1e8 yr lifetime (Micelotta+10b, as
#:   Murga & Wiebe 2022 "Arrh. 2"), RRKM; not a bare-skeleton value, the fast bound (grade D).
CARBON_LOSS = {
    "leger89": ("ilt", {"C": (1, 6.2e15, 7.37), "C2": (2, 3.5e17, 8.49), "C3": (3, 1.5e18, 7.97)}),
    "c60_cage": ("ilt", {"C2": (2, 5.0e19, 10.0)}),
    "micelotta10b": ("rrkm", {"C2": (2, 10.0 * 4.184, 4.6)}),
}


NGC7023_EDGE = Environment("NGC 7023 NW edge (A_V ~ 0)", 4100.0, 1.35e4, 3.9, 470.0, 15000)
ORION_BAR_ATOMIC = Environment("Orion Bar atomic PDR", 2.0e4, 5.0e4, 7.0, 500.0, 40000)


class BaselineModel:
    """pahchem baseline for one parent ("C24", "C54" or "C96").

    ``anion`` is the parent anion's PAHSpecies, needed by the statistical
    neutral sticking (``Params.sticking_neutral = "statistical"``).
    """

    def __init__(self, parent, species, params=Params(), anion=None):
        self.parent, self.species, self.p, self.anion = parent, species, params, anion
        self.nc, self.nh0, self.n_solo = PARENTS[parent]
        self.charges = tuple(sorted(IONISATION[parent]))            # Z with a Table A.1 IP, plus none above
        self.nh_values = tuple(range(0, self.nh0 + 5))
        self._cm = {}
        self._sticking = {}

    def attachment_fraction(self):
        """f of the parent: ``population.attachment_fraction`` under ``Params`` (the size logistic by default), as the
        population gives a molecule of this N_C and Table A.1 EA."""
        from types import SimpleNamespace

        from pycalima.pahchem import population

        th = IONISATION[self.parent]
        return population.attachment_fraction(SimpleNamespace(nc=self.nc, ea_ev=th.get(-1), ionisation_ev=th), self.p)

    def capture_coefficients(self, temperature_k):
        """Electron capture coefficient of each charge [cm^3 s^-1], as the population's (``population.capture_coefficient``):
        BT94 with the neutral's ``sticking_neutral`` and the cation factor of ``Params.capture_model`` (``lab_cold``
        by default), ``capture_scale`` on every charge. Until 2026-11-02 the baseline took sticking 1 on cations
        whatever ``capture_model``."""
        from pycalima.pahchem import population

        return np.array([chemistry.electron_capture(self.nc, z, temperature_k,
                                                    sticking=self.sticking_neutral(temperature_k) if z == 0 else 1.0)
                         * population._capture_factor(self, z, self.p, temperature_k) for z in self.charges])

    def sticking_neutral(self, temperature_k):
        """Electron sticking coefficient of the neutral at gas temperature T.

        A number in ``Params.sticking_neutral`` is used as given. With
        "statistical" it is the parent's ``attachment_fraction`` f times the s_e of
        ``attachment.Attachment`` (with that capture fraction) for the parent and its
        anion (PAHdb modes, EA of Andrews+16 Table A.1), the neutral's internal
        energy being its radiative P(E) under the Draine field at G0 = 1; that
        P(E) changes s_e by < 10 % up to G0 = 1e4 (pahchem-spec.md, M4). Every
        N_H takes the parent's value. Until 2026-11-02 f was ``Params.attachment_fraction`` whatever
        ``attachment_model``.
        """
        s = self.p.sticking_neutral
        if not isinstance(s, str):
            return float(s)
        if s != "statistical":
            raise ValueError(f"sticking_neutral must be a number or 'statistical', not {s!r}")
        if self.anion is None:
            raise ValueError("statistical neutral sticking needs the anion's modes (BaselineModel(anion=...))")
        key = round(float(temperature_k), 6)
        if key not in self._sticking:
            from pycalima.pahchem.attachment import Attachment, grid_excess_cm

            if "neutral_pe" not in self._cm:             # the neutral's radiative P(E) under Draine G0 = 1
                cm = energy_me.CascadeModel(self.species.freqs_cm, scaled_einstein_a(self.species, self.p),
                                            e_max_cm=16.0 * dos.CM_PER_EV, de_cm=10.0, grain_cm=1.0)
                sigma = photo.sigma_uv(self.nc, 0, E_PH, self.p.uv_cross_section)
                _, h = self._absorption(fields.draine1978(g0=1.0), sigma, 0, self.nh0)
                self._cm["neutral_pe"] = (cm.energy_cm, cm.steady_state(E_PH, h, {}).probability)
            akey = ("attachment", grid_excess_cm(temperature_k))     # the grid holds the electrons' Maxwellian
            f = self.attachment_fraction()
            if akey not in self._cm:
                self._cm[akey] = Attachment(self.species, self.anion, IONISATION[self.parent][-1],
                                            chemistry.disc_radius_cm(self.nc), e_excess_cm=akey[1],
                                            capture_fraction=f)
            e_n, p_n = self._cm["neutral_pe"]
            self._sticking[key] = f * self._cm[akey].sticking(temperature_k, e_n, p_n)
        return self._sticking[key]

    # -- per-state molecular inputs ------------------------------------------
    def _modes(self, nh):
        if self.p.modes == "stripped" and nh < self.nh0:
            return self.species.stripped(nh)
        return self.species

    def rf_levels(self, z):
        """Recurrent-fluorescence levels [(E_n cm^-1, f)] of charge z, or () (``RF_SPECTRA``)."""
        name = RF_SPECTRA.get(self.parent, {}).get(z)
        if not self.p.recurrent_fluorescence or name is None:
            return ()
        return tuple((e * dos.CM_PER_EV, f) for e, f in photo.malloci_rf_levels(name, z))

    def _cascade(self, nh, z=None):
        """Cascade of state N_H; with ``z``, including that charge's RF levels."""
        sp = self._modes(nh)
        levels = self.rf_levels(z) if z is not None else ()
        key = (sp.n_modes if self.p.modes == "stripped" else "parent", levels)
        if key not in self._cm:
            cm = energy_me.CascadeModel(sp.freqs_cm, scaled_einstein_a(sp, self.p) * self.p.ir_scale,
                                        e_max_cm=self.p.e_max_ev * dos.CM_PER_EV, de_cm=self.p.de_cm,
                                        electronic=levels)
            E, counts = dos.beyer_swinehart(sp.freqs_cm, cm.energy_cm[-1] + 200.0)
            self._cm[key] = (sp, cm, E, counts, {})
        return self._cm[key]

    def _channel_rate(self, nh, z, name):
        sp, cm, E, counts, cache = self._cascade(nh)
        p = self.p
        if name == "H" and nh > self.nh0:
            e0, ds, sigma = p.h_superh_e0[1 if z > 0 else 0], p.h_superh_ds, nh - self.nh0
        elif name == "H":
            (e0, ds), sigma = (p.h_even, nh) if nh % 2 == 0 else (p.h_odd, 1)
        else:
            (e0, ds), sigma = p.h2, 2
        key = (name, e0, ds, sigma)
        if key not in cache:
            if p.rate_form == "rrkm":
                ts = rates.transition_state_for_entropy(sp.freqs_cm, 3050.0, ds, n_scaled=5)
                k = rates.rrkm(E, counts, ts, e0 * dos.CM_PER_EV, degeneracy=sigma)
            else:   # Andrews+16 Table 2 effective-temperature form (no degeneracy)
                from pycalima.pahchem.andrews16 import H_LOSS_EVEN, H_LOSS_ODD, H2_LOSS, H_LOSS_SUPERH
                ea, dsa = (H_LOSS_SUPERH["Z>0" if z > 0 else "Z<=0"] if nh > self.nh0 else
                           (H_LOSS_EVEN if nh % 2 == 0 else H_LOSS_ODD) if name == "H" else H2_LOSS)
                cache[key] = rates.arrhenius_effective_temperature(sp.freqs_cm, cm.energy_cm,
                                                                   ea * dos.CM_PER_EV, dsa)
                return cache[key]
            cache[key] = k[np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)]
        return cache[key]

    def _channels(self, nh, z):
        ch = {}
        if nh > 0:
            ch["H"] = self._channel_rate(nh, z, "H")
        if nh <= self.nh0 and nh % 2 == 0 and nh > self.n_solo and nh >= 2:
            ch["H2"] = self._channel_rate(nh, z, "H2")
        return ch

    def ionisation_energy(self, z, nh):
        ip = IONISATION[self.parent].get(z)
        return None if ip is None else ip + DELTA_IP * (self.nh0 - nh)

    # -- rates -----------------------------------------------------------------
    def photo_rates(self, env):
        """{(Z, N_H): {"H", "H2", "ion"}} [s^-1] with physical (sink) cascades."""
        field = env.field()
        out = {}
        for z in self.charges:
            sigma = photo.sigma_uv(self.nc, z, E_PH, self.p.uv_cross_section) * self.p.sigma_scale
            for nh in self.nh_values:
                y, h = self._absorption(field, sigma, z, nh)
                ch = self._channels(nh, z)
                rec = dict(self._cascade(nh, z)[1].steady_state(E_PH, h, ch).rates) if ch else {}
                rec["ion"] = photo.ionisation_rate(field, sigma, y, E_PH)
                out[(z, nh)] = rec
        return out

    def _absorption(self, field, sigma, z, nh):
        """(ionisation yield, heating spectrum) of state (Z, N_H)."""
        ip = self.ionisation_energy(z, nh)
        if ip is None or ip >= 13.6:
            y = np.zeros_like(E_PH)
        elif z == 1 and self.p.yield_cation == "wenzel" and self.nc >= 32 and ip < 11.3:
            y = photo.yield_wenzel2020(E_PH, ip, self.nc)
        elif self.p.yield_model == "lepage":
            y = photo.yield_lepage2001(E_PH, ip)
        else:
            y = photo.yield_jochims1996(E_PH, ip)
        heat_y = np.zeros_like(E_PH) if self.p.ionising_photons_heat else y
        return y, photo.heating_spectrum(field, sigma, heat_y, E_PH)

    def carbon_loss_rates(self, env, bare=None):
        """{Z: carbon atoms lost per second} of the bare skeleton (N_H = 0).

        Each fragment of ``CARBON_LOSS[params.carbon_loss]`` is a sink of the
        cascade, so the channels compete with each other and with IR cooling;
        the result is sum_fragment n_C k_fragment. ``bare`` is the PAHSpecies
        of the bare skeleton (e.g. PAHdb C54); if None, ``species.stripped(0)``.
        An RRKM entry builds its transition state by removing the C-C stretch
        nearest 1600 cm^-1 and scaling the 5 lowest modes to its dS. Sources
        and confidence are in the card.
        """
        sp = self.species.stripped(0) if bare is None else bare
        form, fragments = CARBON_LOSS[self.p.carbon_loss]
        if "bare" not in self._cm:
            cm = energy_me.CascadeModel(sp.freqs_cm, scaled_einstein_a(sp, self.p) * self.p.ir_scale,
                                        e_max_cm=self.p.e_max_ev * dos.CM_PER_EV, de_cm=self.p.de_cm)
            E, counts = dos.beyer_swinehart(sp.freqs_cm, cm.energy_cm[-1] + 200.0)
            pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
            k = {}
            for name, (_, a_or_ds, e0) in fragments.items():
                if form == "ilt":
                    k[name] = rates.inverse_laplace_arrhenius(E, counts, a_or_ds, e0 * dos.CM_PER_EV)[pick]
                else:
                    ts = rates.transition_state_for_entropy(sp.freqs_cm, 1600.0, a_or_ds, n_scaled=5)
                    k[name] = rates.rrkm(E, counts, ts, e0 * dos.CM_PER_EV)[pick]
            self._cm["bare"] = (cm, k)
        cm, k = self._cm["bare"]
        field = env.field()
        out = {}
        for z in self.charges:
            sigma = photo.sigma_uv(self.nc, z, E_PH, self.p.uv_cross_section) * self.p.sigma_scale
            _, h = self._absorption(field, sigma, z, 0)
            r = cm.steady_state(E_PH, h, k).rates
            out[z] = float(sum(fragments[name][0] * r[name] for name in fragments))
        return out

    def network(self, photo_rates, env):
        p, T = self.p, env.temperature_k
        net = PAHNetwork(self.charges, self.nh_values)
        Z = np.array(self.charges)[:, None] * np.ones((1, len(self.nh_values)), int)
        NH = np.array(self.nh_values)[None, :] * np.ones((len(self.charges), 1), int)
        get = lambda key: np.array([[photo_rates[(z, n)].get(key, 0.0) for n in self.nh_values]
                                    for z in self.charges])
        top_z, top_nh = max(self.charges), max(self.nh_values)
        net.add("H loss", get("H"), dnh=-1)
        net.add("H2 loss", get("H2"), dnh=-2, h2_released=1)
        net.add("photoionisation", np.where(Z < top_z, get("ion"), 0.0), dz=+1)
        per_z = self.capture_coefficients(T)                               # independent of N_H
        capture = np.repeat(per_z[:, None], len(self.nh_values), axis=1)
        net.add("electron capture", np.where(Z > min(self.charges), capture * env.n_e, 0.0), dz=-1)
        sigma_geo = CORONENE_GEOMETRIC_A2 * self.nc / 24.0
        barrier = np.select([NH == self.nh0, NH == self.nh0 + 2], [0.06, 0.035], 0.0)
        neutral = p.addition_neutral_scale * np.where(
            NH < self.nh0, 0.07 * chemistry.cross_section_rate(T, sigma_geo),
            chemistry.cross_section_rate(T, sigma_geo, barrier))
        can_add = NH < top_nh
        net.add("H addition, cations", np.where((Z > 0) & can_add, p.addition_cation * env.n_h, 0.0), dnh=+1)
        net.add("H addition, neutrals", np.where((Z == 0) & can_add, neutral * env.n_h, 0.0), dnh=+1)
        ad_from = {"dehydrogenated": NH < self.nh0, "all": can_add}[p.associative_detachment]
        net.add("associative detachment", np.where((Z == -1) & ad_from, 7.8e-10 * env.n_h, 0.0), dz=+1, dnh=+1)
        er = chemistry.cross_section_rate(T, p.eley_rideal_a2) * env.n_h
        net.add("Eley-Rideal", np.where(NH > self.nh0, er, 0.0), dnh=-1, h2_released=1)
        return net

    def summary(self, env, photo_rates=None):
        """Observables of the steady state at ``env``."""
        pr = self.photo_rates(env) if photo_rates is None else photo_rates
        net = self.network(pr, env)
        f = net.steady_state()
        fh = np.array(list(net.hydrogenation_fractions(f).values()))
        nh = np.array(self.nh_values)
        fz = net.charge_fractions(f)
        return {"f_dehydrogenated": float(fh[nh < self.nh0].sum()),
                "f_superhydrogenated": float(fh[nh > self.nh0].sum()),
                "mean_nh_over_nh0": float(fh @ nh / self.nh0),
                "f_cation": float(sum(v for z, v in fz.items() if z > 0)),
                "f_anion": float(fz.get(-1, 0.0)),
                "h2_per_pah_per_s": net.h2_formation_rate(f)}


#: One-at-a-time perturbations: name -> Params overrides (low, high).
PERTURBATIONS = {
    "H-loss E0 (even) +-0.5 eV": ({"h_even": (3.91, 12.0)}, {"h_even": (4.91, 12.0)}),
    "H-loss dS +-40 J/K/mol": ({"h_even": (4.41, -28.0)}, {"h_even": (4.41, 52.0)}),
    "H-loss E0 (odd) +-0.5 eV": ({"h_odd": (2.66, 10.0)}, {"h_odd": (3.66, 10.0)}),
    "H2-loss E0 +-0.5 eV": ({"h2": (4.08, 40.0)}, {"h2": (5.08, 40.0)}),
    "superH H-loss E0 +-0.3 eV": ({"h_superh_e0": (1.1, 1.25)}, {"h_superh_e0": (1.7, 1.85)}),
    "rate form: Andrews Arrhenius-T_e": ({"rate_form": "arrhenius_te"}, None),
    "mode set: stripped per state": ({"modes": "stripped"}, None),
    "UV cross section x0.5 / x2": ({"sigma_scale": 0.5}, {"sigma_scale": 2.0}),
    "IR cooling x1 / x3 (RF proxy)": (None, {"ir_scale": 3.0}),
    "recurrent fluorescence off": ({"recurrent_fluorescence": False}, None),
    "ionisation yield: Le Page": ({"yield_model": "lepage"}, None),
    "cation ionisation yield: Jochims": ({"yield_cation": "jochims"}, None),
    "ionising photons heat": ({"ionising_photons_heat": True}, None),
    "electron capture x/6.5 (range)": ({"capture_scale": 1 / 6.5}, {"capture_scale": 1.0}),
    "neutral sticking 1e-2": ({"sticking_neutral": 1e-2}, None),
    "cation H addition x0.5 / x2": ({"addition_cation": 0.7e-10}, {"addition_cation": 2.8e-10}),
    "neutral H addition x/3": ({"addition_neutral_scale": 1 / 3}, {"addition_neutral_scale": 3.0}),
    "Eley-Rideal sigma x/3": ({"eley_rideal_a2": 0.02}, {"eley_rideal_a2": 0.18}),
}
#: Perturbations that leave the photo-rates unchanged (reuse them).
CHEMISTRY_ONLY = {"electron capture x/6.5 (range)", "neutral sticking 1e-2", "cation H addition x0.5 / x2",
                  "neutral H addition x/3", "Eley-Rideal sigma x/3"}


def sensitivity(parent, species, env, base=Params(), keys=None, anion=None):
    """{perturbation: (summary_low, summary_high)} plus "baseline"."""
    model = BaselineModel(parent, species, base, anion)
    base_pr = model.photo_rates(env)
    out = {"baseline": model.summary(env, base_pr)}
    for name in (keys or PERTURBATIONS):
        pair = []
        for over in PERTURBATIONS[name]:
            if over is None:
                pair.append(None)
                continue
            m = BaselineModel(parent, species, replace(base, **over), anion)
            pair.append(m.summary(env, base_pr if name in CHEMISTRY_ONLY else None))
        out[name] = tuple(pair)
    return out
