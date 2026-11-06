"""Collisional rate coefficients for the (Z, N_H) network [cm^3 s^-1].

Multiply by the partner's density to get a rate in s^-1.

- ``thermal_speed``: mean speed sqrt(8 k T / pi m).
- ``cross_section_rate``: sigma v exp(-E_b/kT), e.g. H addition over a
  barrier, or Eley-Rideal abstraction. With sigma = 0.06 A^2 (Mennella et al.
  2012) this gives Andrews et al. (2016) eq. 10, 8.7e-13 sqrt(T/100) n_H.
- ``electron_capture``: Bakes & Tielens (1994) capture of an electron by a
  PAH of charge Z, s_e pi a^2 v_e J~(Z) (Draine & Sutin 1987 J~). Defaults:
  a = 0.9 sqrt(N_C) A and the 0.82 disc factor of Verstraete et al. (1990), as
  in Andrews et al. (2016, eq. A.1).
- ``lab_recombination_factor``: the measured PAH-cation recombination
  coefficient over BT94, as a power law in N_C fitted to the FALP data
  (pah_electron_recombination.csv, N_C 6-16, 300 K).
- ``mutual_neutralisation`` and ``ion_neutral_rate``: ion-PAH collisions
  (docs/physics/pahchem-gas-coupling.md, 2.3): X+ + PAH- at an efficiency
  times the DS87 capture with the ion's reduced mass; X+ + PAH0 at an
  efficiency times Langevin, alpha = 1.75 N_C A^3.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

KB_CGS = 1.380649e-16
AMU_G = 1.66053907e-24
ME_G = 9.1093837e-28
KB_EV = 8.617333262e-5
EV_ERG = 1.602176634e-12
E_STATC = 4.80320471e-10

#: ions that collide with PAHs: name -> (mass [amu], ionisation energy of the neutral atom [eV], NIST ASD)
IONS = {"H+": (1.008, 13.598), "He+": (4.003, 24.587), "C+": (12.011, 11.260), "N+": (14.007, 14.534),
        "O+": (15.999, 13.618), "Si+": (28.086, 8.152), "S+": (32.06, 10.360), "Mg+": (24.305, 7.646),
        "Fe+": (55.845, 7.902), "Na+": (22.990, 5.139), "Ne+": (20.180, 21.565)}
#: metal ions that may form [M-PAH]+ adducts below the charge-transfer threshold (Tielens 2021, Sect. 8.7.3)
ADDUCT_IONS = ("Na+", "Mg+", "Fe+", "Si+")
#: mutual-neutralisation efficiency on the DS87 capture in pahchem's geometry: the Shuman+14 polyatomic law at
#: 300 K for C50H18- (check_pah_ion_rates.py; Hickman 1979 gives 0.80-0.98, Tielens 2021 eq. 8.112 0.46)
MN_EFFICIENCY = 0.58
#: total ion-neutral reaction efficiency on Langevin: anthracene + He+ and C+ at 293 K (Canosa+95, 0.88-0.90)
ION_NEUTRAL_EFFICIENCY = 0.9
#: PAH polarisability per C atom [A^3] for the Langevin rate (spec 2.3; Tielens 2021 Table 6.5 gives ~2.3 at C54)
ALPHA_PER_C_A3 = 1.75


def thermal_speed(temperature_k, mass_g):
    """Mean thermal speed [cm s^-1]."""
    return np.sqrt(8.0 * KB_CGS * np.asarray(temperature_k) / (np.pi * mass_g))


def cross_section_rate(temperature_k, sigma_a2, barrier_ev=0.0, partner_mass_amu=1.00794):
    """sigma v exp(-E_b / kT) [cm^3 s^-1] for a cross section in A^2."""
    v = thermal_speed(temperature_k, partner_mass_amu * AMU_G)
    return sigma_a2 * 1e-16 * v * np.exp(-barrier_ev / (KB_EV * np.asarray(temperature_k)))


def disc_radius_cm(nc):
    """PAH radius 0.9 sqrt(N_C) A used for charging (Bakes & Tielens 1994)."""
    return 0.9e-8 * np.sqrt(nc)


def electron_capture(nc, charge, temperature_k, sticking=1.0, disc_factor=0.82, radius_cm=None):
    """Electron capture coefficient by a PAH of charge Z [cm^3 s^-1]."""
    from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87

    a = disc_radius_cm(nc) if radius_cm is None else radius_cm
    J = J_function_DS87(int(charge), -1, a, float(temperature_k))
    return disc_factor * sticking * np.pi * a**2 * thermal_speed(temperature_k, ME_G) * J


_LAB_FIT = {}


def lab_recombination_fit():
    """(a, b, n) of log10(k_measured / k_BT94) = a + b log10(N_C) over the FALP data at 300 K.

    Every measurement in pah_electron_recombination.csv counts once (the two
    naphthalene values, 3x apart, both). BT94 is ``electron_capture`` with
    sticking 1 at 300 K.
    """
    if not _LAB_FIT:
        import pandas as pd

        from pycalima._paths import get_external_data_path

        t = pd.read_csv(get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
        bt = np.array([electron_capture(n, 1, 300.0) for n in t.n_c])
        b, a = np.polyfit(np.log10(t.n_c.to_numpy(float)), np.log10(t.k_cm3_s.to_numpy() / bt), 1)
        _LAB_FIT.update(a=float(a), b=float(b), n=int(len(t)))
    return _LAB_FIT["a"], _LAB_FIT["b"], _LAB_FIT["n"]


def lab_recombination_factor(nc, cap=None):
    """k_measured / k_BT94 at N_C from ``lab_recombination_fit``, optionally capped at ``cap``.

    The fit spans N_C 6-16 only. Beyond that it is an extrapolation (about
    1.15 at N_C = 100), and ``cap`` bounds it. The factor is T-independent: the data
    are at 300 K.
    """
    a, b, _ = lab_recombination_fit()
    f = 10.0 ** (a + b * np.log10(np.asarray(nc, dtype=float)))
    return np.minimum(f, cap) if cap is not None else f


#: subsets of pah_electron_recombination.csv for ``lab_sticking``. "cold": the photo-ion measurements (FlAPI, VENDAMS), whose
#: ions are near their ground state, as interstellar PAH cations are (IR-cooled long before an electron arrives); the
#: charge-transfer ions of FALP-MS can be internally hot (Biennier+06). "pahs": every PAH (N_C >= 10). Benzene is in neither:
#: its recombination is dissociative, the PAHs' is not.
LAB_STICKING_SUBSETS = ("cold", "pahs")
_LAB_STICKING = {}


def lab_sticking_fit(subset="cold"):
    """(N_s, b, n) of s(N_C) = 1 - exp[-(N_C/N_s)^b], fitted to k_measured / k_BT94 at 300 K over ``subset`` in log10, each
    point weighted by its quoted error. s rises as (N_C/N_s)^b where the data are and saturates at sticking 1 (the molecular
    limit: a captured electron is not re-emitted); the saturation lies beyond the data (N_C <= 16), an extrapolation."""
    if subset not in LAB_STICKING_SUBSETS:
        raise ValueError(f"subset must be one of {LAB_STICKING_SUBSETS}, not {subset!r}")
    if subset not in _LAB_STICKING:
        import pandas as pd
        from scipy.optimize import least_squares

        from pycalima._paths import get_external_data_path

        t = pd.read_csv(get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
        t = t[t.n_c >= 10]
        if subset == "cold":
            t = t[t.method != "FALP-MS"]
        n = t.n_c.to_numpy(float)
        y = np.log10(t.k_cm3_s.to_numpy() / np.array([electron_capture(x, 1, 300.0) for x in n]))
        sig = np.log10(1.0 + t.err_frac.to_numpy())

        def resid(p):
            return (np.log10(-np.expm1(-(n / np.exp(p[0])) ** p[1])) - y) / sig

        fit = least_squares(resid, x0=[np.log(25.0), 2.5], bounds=([np.log(1.0), 0.1], [np.log(1e4), 10.0]))
        _LAB_STICKING[subset] = (float(np.exp(fit.x[0])), float(fit.x[1]), int(n.size))
    return _LAB_STICKING[subset]


def lab_sticking(nc, subset="cold"):
    """Electron sticking on PAH cations from ``lab_sticking_fit`` (a factor on BT94, in (0, 1])."""
    n_s, b, _ = lab_sticking_fit(subset)
    return -np.expm1(-(np.asarray(nc, dtype=float) / n_s) ** b)


def _reduced_mass_g(ion, nc, nh):
    m_x = IONS[ion][0]
    m_pah = 12.011 * nc + 1.008 * nh
    return m_x * m_pah / (m_x + m_pah) * AMU_G


def mutual_neutralisation(nc, nh, charge, ion, temperature_k, efficiency=MN_EFFICIENCY, disc_factor=0.82):
    """X+ + PAH^Z (Z <= -1) -> X + PAH^(Z+1) [cm^3 s^-1]: efficiency x 0.82 pi a^2 v J~(Z, +1) (DS87, attractive)."""
    from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87

    if charge > -1:
        raise ValueError("mutual neutralisation needs a negative PAH")
    a = disc_radius_cm(nc)
    v = np.sqrt(8.0 * KB_CGS * temperature_k / (np.pi * _reduced_mass_g(ion, nc, nh)))
    return efficiency * disc_factor * np.pi * a**2 * v * J_function_DS87(int(charge), 1, a, float(temperature_k))


#: the energy per amu [eV] above which an ion leaves a grain still ionised, (M / m_e) I_H at the Bohr speed (Draine &
#: Salpeter 1979, Sect. III.b)
BOHR_EV_PER_AMU = 13.605693 / 5.48579909e-4


#: the C60 geometrical cross section Moretto-Capelle+04 refer their capture to [cm^2] ("about 35e-16")
C60_GEOMETRIC_CM2 = 35e-16
#: the collision energies [keV per amu] of the Rapp-Francis fit, where capture dominates the C60+ production
CAPTURE_FIT_MAX_KEV_AMU = 50.0


@lru_cache(maxsize=None)
def capture_fit():
    """(A, B) of sqrt(sigma / 1e-16 cm^2) = A - B ln(E / keV per amu), Rapp & Francis (1962)'s form, fitted to the single
    capture of H+, H2+ and H3+ on C60 at equal velocities up to CAPTURE_FIT_MAX_KEV_AMU (Moretto-Capelle+04 Fig. 2, read
    from the figure: ``moretto2004_fig2_capture.csv``)."""
    from pycalima._paths import get_external_data_path

    with open(get_external_data_path("pahchem", "moretto2004_fig2_capture.csv")) as f:
        rows = [line.strip().split(",") for line in f if not line.startswith("#")][1:]
    e = np.array([float(r[1]) for r in rows])
    sig = np.array([float(r[2]) for r in rows])
    keep = e <= CAPTURE_FIT_MAX_KEV_AMU
    slope, a = np.polyfit(np.log(e[keep]), np.sqrt(sig[keep]), 1)
    return float(a), float(-slope)


def capture_efficiency(e_kev_per_amu):
    """sigma_capture / C60_GEOMETRIC_CM2 of H+ + C60 at ``e_kev_per_amu`` (``capture_fit``): measured at 6.7-50 keV per amu
    (0.22-0.07 on the fit), an extrapolation below it: 0.44 at 1, 0.81 at 0.1, 1.29 at 0.01 keV per amu (kT of H+ at
    ~1e8, 1e6, 1e5 K). It reaches the ~2.4 of the classical over-the-barrier capture radius on neutral C60 (Zettergren+02,
    an upper bound for a conducting sphere) near 1e-4 keV per amu, kT at ~1e3 K for H+ and ~1e4 K for O+ (where
    ``capture_efficiency_mean`` gives 2.4); it is not capped."""
    a, b = capture_fit()
    root = np.maximum(a - b * np.log(np.asarray(e_kev_per_amu, dtype=float)), 0.0)
    return root**2 * 1e-16 / C60_GEOMETRIC_CM2


def ds87_barrier_ev(nc, charge):
    """DS87's reduced barrier for a singly charged ion on PAH^Z (Z >= 1) [eV]: theta = nu / (1 + nu^-1/2) e^2 / a, nu = Z
    (their eq. 3.6, the image attraction included)."""
    nu = float(charge)
    return nu / (1.0 + nu**-0.5) * E_STATC**2 / disc_radius_cm(nc) / EV_ERG


def capture_efficiency_mean(nc, nh, charge, ion, temperature_k, n=200, deficit_ev=0.0):
    """``capture_efficiency`` averaged over the ions that reach PAH^Z (Z >= 1), each at the speed of its relative kinetic
    energy E: the Maxwellian flux over DS87's reduced barrier theta (``ds87_barrier_ev``), weight (E - theta) e^(-E / kT);
    for an endothermic transfer (``ion_cation_transfer``'s ``deficit_ev``) over those with E >= deficit_ev only."""
    kt = KB_EV * float(temperature_k)
    theta = ds87_barrier_ev(nc, charge)
    x0 = max(float(deficit_ev) - theta, 0.0)                              # the least kinetic energy above the barrier
    u = kt * np.geomspace(1e-4, 50.0, n)
    x = x0 + u                                                            # kinetic energy above the barrier [eV]
    w = x * np.exp(-u / kt) * u                                           # (E - theta) e^(-E/kT), per d ln u
    eps = capture_efficiency((theta + x) / (_reduced_mass_g(ion, nc, nh) / AMU_G) * 1e-3)
    return float(np.trapezoid(w * eps, np.log(u)) / np.trapezoid(w, np.log(u)))


def ion_cation_transfer(nc, nh, charge, ion, temperature_k, disc_factor=0.82, deficit_ev=0.0):
    """X+ + PAH^Z (Z >= 1) -> X + PAH^(Z+1) in hot gas [cm^3 s^-1]: eps 0.82 pi a^2 v J~(Z, +1) (DS87, repulsive with the
    image attraction), less the ions that cross faster than the Bohr speed and leave still ionised (DS79: arrivals above
    E_B = (M / m_e) I_H, a fraction (1 + x) e^-x, x = E_B / kT). eps is ``capture_efficiency_mean``: the single capture of
    H+ on neutral C60 over its geometrical cross section at the same speed (Moretto-Capelle+04, the only measured analogue;
    no atomic-ion capture on a PAH cation has been measured), measured at 6.7-50 keV per amu and extrapolated below it with
    Rapp & Francis' form (``capture_fit``). Until 2026-11-05 eps was the ion-neutral 0.9 at every speed (pahchem audit, the
    user's choice).

    ``deficit_ev`` > 0: the transfer is endothermic by that much (the atom's IP below the PAH state's threshold; the
    caller's). The ion leaves neutral, so the Coulomb energy it climbed returns to the pair and its kinetic energy at
    infinity E must cover the deficit; the arrivals over the barrier theta (``ds87_barrier_ev``) are a flux
    (E - theta) e^(-E/kT), of which those with E >= deficit are (1 + y) e^-y, y = max(deficit - theta, 0) / kT, and eps is
    their mean (the threshold law, the user's choice of 2026-11-06; endothermic transfers were dropped before)."""
    from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87

    if charge < 1:
        raise ValueError("ion_cation_transfer needs a positive PAH")
    a = disc_radius_cm(nc)
    v = np.sqrt(8.0 * KB_CGS * temperature_k / (np.pi * _reduced_mass_g(ion, nc, nh)))
    eps = capture_efficiency_mean(nc, nh, charge, ion, temperature_k, deficit_ev=deficit_ev)
    x = IONS[ion][0] * BOHR_EV_PER_AMU / (KB_EV * temperature_k)
    rate = (eps * disc_factor * np.pi * a**2 * v * J_function_DS87(int(charge), 1, a, float(temperature_k))
            * (1.0 - (1.0 + x) * np.exp(-x)))
    if deficit_ev > 0.0:
        y = max(float(deficit_ev) - ds87_barrier_ev(nc, charge), 0.0) / (KB_EV * temperature_k)
        rate *= (1.0 + y) * np.exp(-y)
    return rate


def ion_neutral_rate(nc, nh, ion, efficiency=ION_NEUTRAL_EFFICIENCY, alpha_per_c_a3=ALPHA_PER_C_A3):
    """X+ + PAH0 -> products with Delta Z = +1 [cm^3 s^-1]: efficiency x Langevin 2 pi e (alpha/mu)^1/2.

    Whether it happens is decided by the caller (charge transfer needs IP(X) > IP of the PAH; adducts are an
    option below that). For C+ the products are 60 % charge transfer and 40 % carbon accretion (Canosa+95).
    """
    alpha = alpha_per_c_a3 * nc * 1e-24
    return efficiency * 2.0 * np.pi * E_STATC * np.sqrt(alpha / _reduced_mass_g(ion, nc, nh))
