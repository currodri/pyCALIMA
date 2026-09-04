"""
DUST SUBLIMATION

Tools to model the thermal sublimation of dust grains following the
formalism of Guhathakurta & Draine (1989, ApJ 345, 230; "GD89").

The workflow combines the building blocks already present in the
``models`` package:

1. A heating radiation field (Mathis et al. 1983 ISRF scaled by ``G0``,
   or a Kurucz O6V stellar atmosphere geometrically diluted with
   distance) built in the same units used elsewhere in
   ``dust_emission.py``.
2. The bin-averaged absorption cross sections produced by
   ``compute_cross_sections`` (which reads the Draine optical efficiency
   tables handled in ``dust_oppacity.py``) for every ``DustBin`` defined
   in the grain-size JSON configuration (e.g. ``grain_size_distribution.json``).
3. The radiative-equilibrium dust temperature obtained from the
   absorbed/emitted power balance implemented in ``dust_emission.py``.

With the equilibrium grain temperature in hand, the GD89 sublimation rate
gives the rate at which the grain radius shrinks and the associated
sublimation timescale.

GD89 model
----------
The number of monomers lost from a grain per unit time is

    dN/dt = - 4*pi*a^2 * J(T_d) * f_corr(N, T_d)

where ``J(T_d)`` is the isolated (canonical) sublimation flux per unit
surface area [cm^-2 s^-1],

    J(T_d) = alpha_N * nu_0 * exp(-B(N) / (k_B * T_d)),

``B(N) = B_inf - sigma * N^(-1/3)`` is the size-corrected binding energy
per monomer (smaller grains are less tightly bound), ``alpha_N`` is the
sticking/evaporation coefficient and ``nu_0`` is the prefactor that
absorbs the lattice vibration frequency and the surface monomer density
so that ``J`` already carries units of cm^-2 s^-1.

``f_corr`` is the GD89 microcanonical correction factor that accounts for
the finite heat reservoir of the grain (the grain cools as each monomer
leaves); it is expressed through ratios of Gamma functions and reduces to
unity for large grains.

Each evaporated monomer removes a volume ``V_mon = m_mon / rho``, so the
grain radius shrinks at

    da/dt = (V_mon / (4*pi*a^2)) * dN/dt = - V_mon * J(T_d) * f_corr,

and the sublimation timescale of a grain of radius ``a`` is

    tau_sub = a / |da/dt|.

By: Curro Rodriguez (currodri@gmail.com)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy import special

from pycalima.models.grain_size_config import get_repo_root, load_grain_size_config
from pycalima.models.dust_radiation.dust_emission import (
    compute_cross_sections,
    interpolate_cross_sections,
    absorbed_power,
    emitted_power,
    modified_mmp83_radiation_field
)

# Physical constants (cgs)
_AMU     = 1.66053906660e-24      # [g] atomic mass unit
_RSUN    = 6.957e10              # [cm] solar radius
_PC      = 3.0856775814913673e18  # [cm] parsec
_YR      = 3.1556952e7            # [s] Julian year
_KB      = 1.380649e-16             # [erg K^-1] Boltzmann constant
_CLIGHT  = 2.99792458e10 # [cm/s] - Speed of light
_HPLANCK = 6.6260755e-27 # [erg s] - Planck constant

# Modelling constants
alpha_N = 0.1   # Sticking coefficient for monomer evaporation (GD89, section 3.1)
GD89_PARAMS = {
    'graphite': {
        'mass_amu': 12.011,  # [amu] carbon atom
        'rho': 2.24,  # [g cm^-3]
        'B_saturation_temperature_K': 81200.0,
        'surface_free_energy_K': 20000.0,
        'evaporation_rate_prefactor': 4.6e30,
        'Debye_temperature_K': 420.0
    },
    'silicate': {
        'mass_amu': 140.69,  # Mg2SiO4 (forsterite monomer, consistent with GD89 section 3.1)
        'rho': 3.5,   # [g cm^-3] Bulk density of silicate (GD89 uses rho = 3.5 g/cm^3)
        'B_saturation_temperature_K': 68100.0,
        'surface_free_energy_K': 20000.0,
        'evaporation_rate_prefactor': 7e30,
        'Debye_temperature_K': 470.0
    }
}

_EXTERNAL_DATA_DIR = os.path.join(str(get_repo_root()), 'external_data')

def _number_of_atoms(a_cm, material):
    """Return the total number of atoms in a cluster of radius a_cm.

    Following Guhathakurta & Draine (1989) Section 3.1 & Figure 4:
    - For graphite, we use the physical volume equation with bulk density rho = 2.24 g/cm^3
      and C atom mass: N_atoms = (4/3) * pi * a^3 * rho / m_C = 0.470422 * a_angstrom^3.
      While GD89 mentions N = 2.0 * a_angstrom^3 in the text, their physical calculations and
      temperature distributions correspond much closer to this physical volume equation.
    - For silicate, N_atoms = 0.44145 * a_angstrom^3. While GD89 mentions 
      N = 1.22 * a_angstrom^3 in the text, their physical calculations and 
      Figure 4 (where a 5.5 A grain carries ~73 atoms) correspond to the 
      actual physical volume equation with bulk density rho = 3.5 g/cm^3 and 
      mean atomic mass mu = 140.69 / 7 = 20.1 m_p (Mg2SiO4 forsterite):
      N_atoms = (4/3) * pi * a^3 * rho / (20.1 * m_p) = 0.44145 * a_angstrom^3
    """
    a_angstrom = a_cm * 1e8
    if material == 'graphite':
        return 0.470422 * a_angstrom ** 3
    elif material == 'silicate':
        return 0.44145 * a_angstrom ** 3
    else:
        raise ValueError(f"Unknown material: {material}")


def _number_of_monomers(a_cm, material):
    """Return the number of monomers (formula units) in the grain of radius a_cm.

    For graphite, monomer behaves as a single carbon atom (N_monomers = N_atoms).
    For silicate (Mg2SiO4), there are 7 atoms per formula unit (monomer).
    """
    N_atoms = _number_of_atoms(a_cm, material)
    if material == 'graphite':
        return N_atoms
    elif material == 'silicate':
        return N_atoms / 7.0
    else:
        raise ValueError(f"Unknown material: {material}")


def isolated_sublimation_rate(material, a_cm, Td):
    """Isolated (canonical) sublimation flux J of GD89 (eq. 3.11).

    Parameters
    ----------
    material : str
        ``'graphite'`` or ``'silicate'`` (taken from the JSON ``composition``).
    a_cm : float
        Grain radius in cm (sets the number of monomers ``N``).
    Td : float
        Grain temperature in K.

    Returns
    -------
    float
        Number of monomers leaving the grain per unit surface area and
        time, in cm^-2 s^-1. The binding energy ``B`` is reduced for small
        grains via the surface free-energy term ``sigma * N^(-1/3)`` 
        where N is the number of monomers (for silicate, olivine units).
    """
    params = GD89_PARAMS[material]
    # For graphite, N_binding = N_atoms. For silicate, N_binding = N_monomers.
    N_binding = _number_of_monomers(a_cm, material)
    # Size-corrected binding energy per monomer, expressed as B/k_B in K.
    B_over_kB = params['B_saturation_temperature_K'] - (params['surface_free_energy_K'] * N_binding ** (-1. / 3.))
    exp_factor = np.exp(-B_over_kB / Td)
    # prefactor [cm^-2 s^-1] already folds in nu_0 and the surface monomer density.
    return params['evaporation_rate_prefactor'] * exp_factor * alpha_N

def silicate_heat_capacity(Td):

    # Return in erg/cm3/K the heat capacity of silicate grains as a function of temperature
    if Td <= 50.:
        return 1.4e3 * Td**2.
    elif Td <= 150.:
        return 2.2e4 * Td**1.3
    elif Td <= 500.:
        return 4.8e5 * Td**0.68
    else:
        return 3.41e7

def get_enthalpy(material, a_cm, Td):
    """Internal (thermal) energy stored in the grain at temperature Td.

    Returns the total enthalpy in erg.  For graphite the DL01 formula
    gives H directly in erg per C atom (Chase 1985 fit, valid 0–6000 K);
    for silicate the bulk heat capacity [erg/cm^3/K] is integrated over
    the grain volume.  The ``(1 - 2/N)`` factor removes the contribution
    of the translational/rotational degrees of freedom retained when a
    single monomer leaves the lattice.
    """
    N_atoms = _number_of_atoms(a_cm, material)  # Total number of atoms in the grain
    if material == 'graphite':
        H_per_atom = (4.15e-22 * Td**3.3) / (1. + 6.51e-3 * Td + 1.5e-6 * Td**2. + 8.3e-7 * Td**2.3)  # [erg/atom]
        H_total = (1. - 2. / N_atoms) * N_atoms * H_per_atom  # [erg]
        return H_total
    elif material == 'silicate':
        T_range = np.linspace(0.1, Td, 500)
        C_sil = np.array([silicate_heat_capacity(T) for T in T_range]) # [erg/cm3/K]
        C_integral = np.trapezoid(C_sil, T_range) # [erg/cm3]
        V = (4.0 / 3.0) * np.pi * a_cm ** 3  # [cm3]
        H_total = (1. - 2. / N_atoms) * V * C_integral # [erg]
        return H_total
    else:
        raise ValueError(f"Unknown material: {material}")


def correction_factor(material, a_cm, Td):
    """GD89 microcanonical correction factor f_corr (dimensionless).

    Accounts for the finite internal energy of the grain, which cools as
    each monomer evaporates. Evaluated through ratios of Gamma functions
    (GD89); the ratio is computed in log-space with ``gammaln`` to avoid
    the ``inf/inf`` overflow that arises for hot grains, where the Gamma
    arguments become very large. Returns 0 when the grain internal energy
    is too low for the expansion to be defined (no sublimation).
    """
    params = GD89_PARAMS[material]
    N_atoms = _number_of_atoms(a_cm, material)  # Total number of atoms in the grain (thermal reservoir)
    f = 3. * N_atoms - 6.                          # vibrational degrees of freedom
    # For graphite, N_binding = N_atoms. For silicate, N_binding = N_monomers.
    N_binding = _number_of_monomers(a_cm, material)
    B = (params['B_saturation_temperature_K'] - params['surface_free_energy_K'] * N_binding ** (-1. / 3.))  # [K]
    b = B / (0.75 * params['Debye_temperature_K'])           # dimensionless
    U = get_enthalpy(material, a_cm, Td)                       # [erg]
    m = U / (0.75 * _KB * params['Debye_temperature_K'])      # dimensionless (k_B in erg/K)
    gamma = m / f
    # Microcanonical density-of-states ratio rho(U-B)/rho(U) for f harmonic
    # modes carrying m quanta (RRK/Einstein counting), multiplied by the
    # canonical reference correction ((1+gamma)/gamma)**b. Arguments of the
    # four Gamma functions below (note gamma*f == m); this reduces to
    # f_corr -> 1 in the large-grain (canonical) limit.
    args = (m + 1., m + f - b, m - b + 1., m + f)
    if gamma <= 0.0 or any(arg <= 0.0 for arg in args):
        # Grain too cold / not enough internal energy: rate is negligible.
        return 0.0
    # log f_corr = b*ln((1+gamma)/gamma) + lnGamma(m+1) + lnGamma(m+f-b)
    #              - lnGamma(m-b+1) - lnGamma(m+f)
    log_corr = (b * np.log((1. + gamma) / gamma)
                + special.gammaln(args[0]) + special.gammaln(args[1])
                - special.gammaln(args[2]) - special.gammaln(args[3]))
    return np.exp(log_corr)


def sublimation_rate(material, a_cm, Td):
    """Monomer loss rate dN/dt of the grain in s^-1.

    Combines the isolated GD89 flux [cm^-2 s^-1], the microcanonical
    correction factor, and the grain surface area [cm^2] to give the
    number of monomers leaving the grain per second.
    """
    grain_area = 4.0 * np.pi * a_cm ** 2  # [cm^2]
    return grain_area * isolated_sublimation_rate(material, a_cm, Td) * correction_factor(material, a_cm, Td)


def radius_loss_rate(material, a_cm, Td):
    """Grain radius shrink rate |da/dt| in cm s^-1.

    Each evaporated monomer removes a volume ``V_mon = m_mon / rho``, so
    ``da/dt = V_mon * (dN/dt) / (4*pi*a^2)``.
    """
    params = GD89_PARAMS[material]
    m_mon = params['mass_amu'] * _AMU      # [g]   monomer mass
    V_mon = m_mon / params['rho']          # [cm^3] volume removed per monomer
    grain_area = 4.0 * np.pi * a_cm ** 2   # [cm^2]
    dN_dt = sublimation_rate(material, a_cm, Td)  # [s^-1]
    return V_mon * dN_dt / grain_area      # [cm s^-1]


def mass_loss_rate(material, a_cm, Td):
    """Grain mass loss rate |dm/dt| in g s^-1.

    Derived from the radius loss rate via

        |dm/dt| = 4 pi a^2 rho |da/dt|

    where ``rho`` is the bulk grain density [g cm^-3]. This is the rate at
    which the grain mass decreases due to thermal sublimation.
    """
    params = GD89_PARAMS[material]
    rho = params['rho']                                     # [g cm^-3]
    grain_area = 4.0 * np.pi * a_cm ** 2                   # [cm^2]
    return grain_area * rho * radius_loss_rate(material, a_cm, Td)  # [g s^-1]


def sublimation_timescale(material, a_cm, Td):
    """GD89 sublimation timescale tau_sub = a / |da/dt| in seconds.

    Following GD89 (Section 3.4), the lifetime of a grain of radius a against
    sublimation is defined as tau_sub = a / |da/dt|, where |da/dt| is the
    grain radius contraction/shrink rate.

    Returns ``np.inf`` where the sublimation rate underflows to zero
    (cold grains effectively never sublimate).
    """
    da_dt = radius_loss_rate(material, a_cm, Td)  # [cm s^-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(da_dt > 0.0, a_cm / da_dt, np.inf)
    return tau


# ---------------------------------------------------------------------------
# Radiation fields
# ---------------------------------------------------------------------------
def mathis_radiation_field_spectrum(G0=1.0, lambda_min_micron=0.08,
                                    lambda_max_micron=12.0, n_points=600,
                                    use_gd89_isrf=False):
    """Full Mathis et al. (1983) ISRF (UV + optical) scaled by ``G0``.

    If ``use_gd89_isrf`` is True, it loads the exact digitized GD89 paper's
    Mathis et al. (1983) ISRF from ``external_data/Mathis_ISRF_GD89.csv`` (which has 
    been scaled such that u_UV = 4.0e-14 erg/cm3 for G0=1.0).

    Otherwise, it uses ``modified_mmp83_radiation_field`` which includes both the UV
    power-law component and the three diluted stellar blackbodies
    (T = 3000, 4000, 7500 K) from the original Mathis et al. (1983) field.

    The field is returned in the same convention used by
    ``dust_emission.absorbed_power``: a spectral flux ``4*pi*J_lambda`` in
    erg s^-1 cm^-2 cm^-1, so that ``trapz(field * C_abs, wavelength)`` is
    the absorbed power in erg s^-1.

    Parameters
    ----------
    G0 : float
        Scaling factor relative to the standard Mathis ISRF.
    lambda_min_micron, lambda_max_micron : float
        Wavelength range in micron.
    n_points : int
        Number of logarithmically spaced wavelength samples.
    use_gd89_isrf : bool
        If True, load exact digitized GD89 paper's Mathis et al. (1983) ISRF.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Wavelengths in cm and the spectral flux in erg s^-1 cm^-2 cm^-1.
    """
    if use_gd89_isrf:
        csv_path = os.path.join(_EXTERNAL_DATA_DIR, 'Mathis_ISRF_GD89.csv')
        if os.path.exists(csv_path):
            data = np.loadtxt(csv_path, delimiter=',')
            # Column 0: wavelength in micron -> convert to cm
            wav_cm = data[:, 0] * 1e-4
            # Column 1: energy density u_lambda in erg/cm^3/micron.
            # Convert to spectral flux 4pi*J_lambda = u_lambda * c
            # col1 [erg/cm3/micron] * 1e4 [micron/cm] = u_lambda [erg/cm3/cm]
            u_lam = data[:, 1] * 1e4
            # Scale so that integrated UV density <= 2460 A is 4.0e-14 * G0
            # Standard CSV has UV integral = 1.0332e-14 erg/cm3.
            scale_fac = (4.0e-14 / 1.0332e-14) * G0
            field = u_lam * scale_fac * _CLIGHT
            return wav_cm, field

    wav_cm = np.logspace(np.log10(lambda_min_micron * 1e-4),
                         np.log10(lambda_max_micron * 1e-4), n_points)
    # modified_mmp83_radiation_field returns lambda*u_lambda [erg/cm3].
    # 4pi*J_lambda = u_lambda * c = (lambda*u_lambda / lambda) * c
    lam_ulam = modified_mmp83_radiation_field(wav_cm)  # [erg/cm3]
    field = lam_ulam * _CLIGHT / wav_cm                # [erg/s/cm2/cm]
    return wav_cm, G0 * field


def _load_o6v_intensity():
    """Load the Kurucz O6V (T_eff = 40000 K) surface intensity table.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Wavelengths in cm and the surface specific intensity I_lambda in
        erg s^-1 cm^-2 cm^-1 sr^-1.
    """
    # File layout: two '#'-commented header lines, a 'col1 col2' label row,
    # then two columns: wavelength [nm] and surface intensity
    # I_lambda [erg s^-1 cm^-2 nm^-1 sr^-1].
    data = np.loadtxt(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_40000'), skiprows=3)
    wav_nm = data[:, 0]
    intensity_per_nm = data[:, 1]
    wav_cm = wav_nm * 1e-7
    intensity_per_cm = intensity_per_nm * 1e7  # nm^-1 -> cm^-1
    order = np.argsort(wav_cm)
    return wav_cm[order], intensity_per_cm[order]


def o6v_radiation_field_spectrum(distance_pc, star_radius_rsun=10.0):
    """Geometrically diluted O6V stellar radiation field at ``distance_pc``.

    The diluted field is returned as ``4*pi*J_lambda`` (erg s^-1 cm^-2
    cm^-1), consistent with ``dust_emission.absorbed_power``. The dilution
    factor ``W = 0.5 * (1 - sqrt(1 - (R/d)^2))`` is the exact value for a
    uniformly bright stellar disc and reduces to ``(R / 2d)^2`` far from
    the star.

    Parameters
    ----------
    distance_pc : float
        Distance from the star in parsec.
    star_radius_rsun : float
        Stellar radius in solar radii.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Wavelengths in cm and the spectral flux in erg s^-1 cm^-2 cm^-1.
    """
    wav_cm, intensity = _load_o6v_intensity()
    R = star_radius_rsun * _RSUN
    d = distance_pc * _PC
    ratio = R / d
    if ratio >= 1.0:
        raise ValueError('Distance must be larger than the stellar radius.')
    dilution = 0.5 * (1.0 - np.sqrt(1.0 - ratio ** 2))
    field = 4.0 * np.pi * dilution * intensity
    return wav_cm, field


# ---------------------------------------------------------------------------
# Equilibrium temperature
# ---------------------------------------------------------------------------
def _solve_equilibrium_temperature(wav_field, field, C_abs_field,
                                   wav_em, C_abs_em, Tmin=2.7, Tmax=4000.0):
    """Solve the radiative energy balance for the grain temperature.

    Uses the absorbed/emitted power building blocks from
    ``dust_emission.py``. The bracket is extended up to ``Tmax`` (default
    4000 K) so that grain temperatures relevant for sublimation can be
    found, unlike the default helper which caps at 800 K.
    """
    p_abs = absorbed_power(wav_field, field, C_abs_field)

    def balance(T):
        return p_abs - emitted_power(T, wav_em, C_abs_em)

    f_min = balance(Tmin)
    f_max = balance(Tmax)
    if np.sign(f_min) == np.sign(f_max):
        # No root in range: either negligible heating or extreme heating.
        return Tmin if f_min < 0 else Tmax

    result = root_scalar(balance, bracket=[Tmin, Tmax])
    if result.converged:
        return result.root
    raise RuntimeError('Failed to find equilibrium dust temperature.')


def _interp_cross_section(target_wav_cm, wav_xs_cm, C_abs):
    """Interpolate an absorption cross section onto ``target_wav_cm``.

    ``compute_cross_sections`` returns wavelengths in decreasing order, so
    both arrays are sorted ascending before interpolation.
    """
    order = np.argsort(wav_xs_cm)
    return np.interp(target_wav_cm, wav_xs_cm[order], C_abs[order])


# ---------------------------------------------------------------------------
# Per-bin computation
# ---------------------------------------------------------------------------
def compute_bin_temperature(a0_cm, wav_xs_cm, C_abs, wav_field, field,
                            wav_em=None, Tmax=4000.0):
    """Compute the equilibrium dust temperature for a single bin.

    Parameters
    ----------
    a0_cm : float
        Representative grain radius in cm (used only for bookkeeping).
    wav_xs_cm, C_abs : np.ndarray
        Bin-averaged absorption cross section sampled on ``wav_xs_cm``.
    wav_field, field : np.ndarray
        Incident radiation field grid (cm) and spectral flux.
    wav_em : np.ndarray, optional
        Emission wavelength grid in cm. Defaults to 0.1-1000 micron.
    Tmax : float
        Upper bound for the temperature solver in K.
    """
    if wav_em is None:
        wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    C_abs_field = _interp_cross_section(wav_field, wav_xs_cm, C_abs)
    C_abs_em = _interp_cross_section(wav_em, wav_xs_cm, C_abs)
    return _solve_equilibrium_temperature(wav_field, field, C_abs_field,
                                          wav_em, C_abs_em, Tmax=Tmax)


def _load_dust_bins(config_path=None):
    """Return the non-PAH DustBin metadata from the JSON configuration."""
    cfg = load_grain_size_config(config_path=config_path)
    return [dict(b) for b in cfg['bins'] if not b['is_pah']]


def _prepare_bin_cross_sections(dust_bins):
    """Pre-compute the bin-averaged cross sections and grain sizes.

    Returns a list of dicts holding the bin id, composition, central grain
    size (cm), and the wavelength/absorption arrays.
    """
    prepared = []
    for meta in dust_bins:
        bin_id = meta['id']
        a0_cm, wav_cm, _C_sca, C_abs, _C_rp = compute_cross_sections(bin_id, do_average=True)
        prepared.append({
            'id': bin_id,
            'material': str(meta['composition']).lower(),
            'a0_cm': float(a0_cm),
            'wav_cm': np.asarray(wav_cm, dtype=float),
            'C_abs': np.asarray(C_abs, dtype=float),
        })
    return prepared


# ---------------------------------------------------------------------------
# Stochastic heating: grain temperature probability distribution
# ---------------------------------------------------------------------------
# Small grains (and grains in strong radiation fields) are not in radiative
# equilibrium: each absorbed photon produces a temperature spike followed by
# radiative cooling, so the grain visits a *distribution* of temperatures.
# We compute this distribution with the transition-matrix method of
# Guhathakurta & Draine (1989, "GD89") as reformulated by Camps et al.
# (2015, A&A 580, A87; section 4.3, the SKIRT implementation).
#
# The grain internal energy U is discretised into ``n_bins`` bins with
# representative energies U_i and temperatures T_i = T(U_i). The occupation
# probabilities P_i obey the steady-state master equation built from a
# transition matrix A_{f,i} (rate s^-1 of jumping from bin i to bin f):
#
#   * Heating (f > i): absorption of a photon of energy U_f - U_i. The rate
#     per unit photon energy is
#         R(E) = C_abs(lambda) * [4 pi J_lambda](lambda) * lambda^3 / (h c)^2
#     [s^-1 erg^-1], with lambda = h c / E, and the rate into destination
#     bin f is the integral of R over that bin's photon-energy interval.
#   * Cooling (GD89 approximation): a grain only relaxes to the adjacent
#     lower bin, at the rate set by its thermal emission,
#         A_{i-1,i} = P_emit(T_i) / (U_i - U_{i-1}).
#
# Because down-transitions are single-step, GD89 give an O(N^2) forward
# recursion for the un-normalised probabilities (balance of the net upward
# flux across each bin boundary against the single downward flux):
#
#   A_{f-1,f} P_f = sum_{i<f} P_i * sum_{j>=f} A_{j,i}
#
# solved with P_0 = 1 and finally normalised to sum_i P_i = 1.

_ENERGY_GRID_CACHE = {}
_HC = _HPLANCK * _CLIGHT  # [erg cm] Planck constant times speed of light


def _silicate_heat_capacity_vec(T):
    """Vectorised silicate heat capacity in erg cm^-3 K^-1.

    Same piecewise fit as ``silicate_heat_capacity`` but evaluated on an
    array of temperatures.
    """
    T = np.asarray(T, dtype=float)
    return np.where(T <= 50., 1.4e3 * T ** 2.,
           np.where(T <= 150., 2.2e4 * T ** 1.3,
           np.where(T <= 500., 4.8e5 * T ** 0.68, 3.41e7)))


def _energy_temperature_interpolators(material, a_cm):
    """Return cached (U_of_T, T_of_U) interpolators for the grain.

    ``U(T)`` is the total vibrational (thermal) internal energy of the
    grain in erg. For graphite it uses the carbon enthalpy fit of Draine &
    Li (2001) summed over the ``N`` carbon atoms; for silicate it is the
    grain volume times the integral of the heat capacity. Unlike
    ``get_enthalpy`` (used for sublimation) it omits the ``(1 - 2/N)``
    departing-monomer correction, since here we want the full heat content
    that buffers stochastic heating.

    Returns
    -------
    (callable, callable)
        ``U_of_T(T_K) -> U_erg`` and ``T_of_U(U_erg) -> T_K``, both based on
        ``np.interp`` over a fine, monotonic reference grid.
    """
    key = (material, float(a_cm))
    cached = _ENERGY_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    T_ref = np.logspace(np.log10(0.1), np.log10(1.0e4), 4000)
    N_atoms = _number_of_atoms(a_cm, material)
    factor = 1. - 2. / N_atoms
    if material == 'graphite':
        N = _number_of_monomers(a_cm, material)  # number of C atoms
        H_per_atom = (4.15e-22 * T_ref ** 3.3) / (
            1. + 6.51e-3 * T_ref + 1.5e-6 * T_ref ** 2. + 8.3e-7 * T_ref ** 2.3)  # [erg/atom]
        U_ref = N * H_per_atom * factor  # [erg]
    elif material == 'silicate':
        V = (4.0 / 3.0) * np.pi * a_cm ** 3  # [cm^3]
        C_vec = _silicate_heat_capacity_vec(T_ref)  # [erg cm^-3 K^-1]
        # Cumulative integral of the heat capacity from T_ref[0] (the
        # missing 0 -> 0.1 K contribution is negligible).
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (C_vec[1:] + C_vec[:-1]) * np.diff(T_ref))))
        U_ref = V * cum * factor  # [erg]
    else:
        raise ValueError(f"Unknown material: {material}")

    # Guarantee strict monotonicity for the inverse interpolation.
    U_ref = np.maximum.accumulate(U_ref)

    def U_of_T(T):
        return np.interp(T, T_ref, U_ref)

    def T_of_U(U):
        return np.interp(U, U_ref, T_ref)

    _ENERGY_GRID_CACHE[key] = (U_of_T, T_of_U)
    return U_of_T, T_of_U


def internal_energy(material, a_cm, T):
    """Total vibrational internal energy U(T) of the grain in erg."""
    U_of_T, _ = _energy_temperature_interpolators(material, a_cm)
    return U_of_T(T)


def _emitted_power_fast(T, wav_em, C_abs_em):
    """Vectorised thermal emission power P_emit(T) in erg s^-1.

    Equivalent to ``dust_emission.emitted_power`` but evaluates the Planck
    function on the whole wavelength array at once (the cooling rates need
    one call per energy bin, so the loop-based version is too slow).
    """
    x = (_HC / wav_em) / (_KB * T)
    x = np.clip(x, 0.0, 700.0)
    B_lambda = (2. * _HPLANCK * _CLIGHT ** 2 / wav_em ** 5) / np.expm1(x)  # [erg s^-1 cm^-2 cm^-1 sr^-1]
    return 4. * np.pi * np.trapezoid(C_abs_em * B_lambda, x=wav_em)


def temperature_distribution(material, a_cm, wav_xs_cm, C_abs,
                             wav_field, field, n_bins=128,
                             T_min=2.0, T_max=None, wav_em=None,
                             return_energy=False):
    """Stochastic grain temperature distribution (GD89 / Camps et al. 2015).

    Parameters
    ----------
    material : str
        ``'graphite'`` or ``'silicate'`` (from the JSON ``composition``).
    a_cm : float
        Grain radius in cm.
    wav_xs_cm, C_abs : np.ndarray
        Bin-averaged absorption cross section [cm^2] on grid ``wav_xs_cm`` [cm].
    wav_field, field : np.ndarray
        Heating radiation field grid [cm] and spectral flux ``4*pi*J_lambda``
        [erg s^-1 cm^-2 cm^-1], as returned by the radiation-field helpers.
    n_bins : int
        Number of internal-energy bins.
    T_min : float
        Lower temperature bound of the energy grid in K.
    T_max : float, optional
        Upper temperature bound in K. If ``None`` it is derived from the
        equilibrium temperature plus several maximum-energy photons so that
        the single- and multi-photon spikes are captured.
    wav_em : np.ndarray, optional
        Emission wavelength grid [cm]. Defaults to 0.1-1000 micron.
    return_energy : bool
        If ``True`` also return the representative bin energies [erg].

    Returns
    -------
    (np.ndarray, np.ndarray[, np.ndarray])
        ``T_i`` [K], ``P_i`` (normalised probabilities), and optionally the
        bin energies ``U_i`` [erg].
    """
    # 1. Sort the field ascending in wavelength and resample the cross section.
    order = np.argsort(wav_field)
    wav_f = np.asarray(wav_field, dtype=float)[order]
    fld = np.asarray(field, dtype=float)[order]
    C_abs_f = _interp_cross_section(wav_f, wav_xs_cm, C_abs)

    if wav_em is None:
        wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4
    C_abs_em = _interp_cross_section(wav_em, wav_xs_cm, C_abs)

    U_of_T, T_of_U = _energy_temperature_interpolators(material, a_cm)

    # 2. Per-photon-energy heating rate R(E) [s^-1 erg^-1] and its cumulative
    #    integral G(E) = int_0^E R dE' [s^-1], used to integrate the heating
    #    contribution over each destination bin's photon-energy interval.
    E_phot = _HC / wav_f                                   # [erg]
    R_E = C_abs_f * fld * wav_f ** 3 / _HC ** 2            # [s^-1 erg^-1]
    idx = np.argsort(E_phot)
    E_phot = E_phot[idx]
    R_E = R_E[idx]
    G_cum = np.concatenate(([0.0], np.cumsum(0.5 * (R_E[1:] + R_E[:-1]) * np.diff(E_phot))))
    E_max_field = E_phot[-1]

    def G(E):
        # int_0^E R dE'; 0 below the grid, total above it (captures the tail).
        return np.interp(E, E_phot, G_cum, left=0.0, right=G_cum[-1])

    # 3. Build the internal-energy grid.
    U_min = float(U_of_T(T_min))
    if T_max is None:
        T_eq = _solve_equilibrium_temperature(wav_f, fld, C_abs_f, wav_em, C_abs_em)
        U_eq = float(U_of_T(T_eq))
        U_top = max(U_eq, U_min) + 5.0 * E_max_field
        # Make sure the grid spans at least a couple of maximum-energy photons
        # above the floor so the single-photon spikes are resolved.
        U_top = max(U_top, U_min + 2.0 * E_max_field)
    else:
        # Explicit range requested (used by the adaptive SKIRT heuristic): the
        # grid must respect [T_min, T_max] tightly, with a tiny guard so the
        # logspace edges are strictly increasing.
        U_top = float(U_of_T(T_max))
        if U_top <= U_min:
            U_top = U_min + max(1e-30, 1e-6 * abs(U_min))

    # Build the energy grid using logspace in *temperature* rather than energy.
    # A logspace-U grid wastes ~80% of bins below 500 K (where U(T) is a steep
    # power law) and leaves only a handful of bins in the 2000-4000 K range
    # where sublimation rates are non-negligible. Logspace-T gives equal
    # resolution per decade of T throughout, improving high-T coverage ~6×.
    T_top = float(T_of_U(U_top))
    T_floor = max(T_min, 0.1)
    T_top = max(T_top, T_floor * 2.0)
    T_edges = np.logspace(np.log10(T_floor), np.log10(T_top), n_bins + 1)
    U_edges = np.array([float(U_of_T(t)) for t in T_edges])
    # Enforce strict monotonicity (handles the U_of_T plateau near T=0).
    U_edges = np.maximum.accumulate(U_edges)
    U_lo = U_edges[:-1]
    U_hi = U_edges[1:].copy()
    U_hi[-1] = np.inf  # top bin collects the high-energy tail (photon conservation)
    U_cent = 0.5 * (U_edges[:-1] + U_edges[1:])   # arithmetic bin centres [erg]
    T_cent = T_of_U(U_cent)                        # bin temperatures [K]

    # 4. Heating transition matrix A_heat[f, i] for f > i [s^-1].
    A_heat = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        lo = np.clip(U_lo - U_cent[i], 0.0, None)  # photon energy to reach bin lower edge
        hi = U_hi - U_cent[i]                       # ... upper edge (inf for top bin)
        rate = G(hi) - G(lo)
        rate[:i + 1] = 0.0                          # only up-transitions
        A_heat[:, i] = rate

    # 5. Cooling rates A_{i-1,i} = P_emit(T_i) / (U_i - U_{i-1}) [s^-1].
    cool = np.zeros(n_bins)
    for f in range(1, n_bins):
        P_emit = _emitted_power_fast(T_cent[f], wav_em, C_abs_em)  # [erg s^-1]
        cool[f] = P_emit / (U_cent[f] - U_cent[f - 1])

    # 6. GD89 forward recursion for the un-normalised probabilities.
    #    B_cum[f, i] = sum_{j>=f} A_heat[j, i] (reverse cumulative sum).
    B_cum = np.cumsum(A_heat[::-1, :], axis=0)[::-1, :]
    P = np.zeros(n_bins)
    P[0] = 1.0
    for f in range(1, n_bins):
        if cool[f] <= 0.0:
            P[f] = 0.0
            continue
        P[f] = np.dot(B_cum[f, :f], P[:f]) / cool[f]
        # For large grains the distribution is a sharp peak at equilibrium and
        # the un-normalised probabilities span a huge dynamic range; rescale
        # on the fly to avoid floating-point overflow (the final result is
        # normalised, so a uniform rescaling is harmless).
        if P[f] > 1e290:
            P[:f + 1] /= P[f]

    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total > 0.0:
        P /= total

    if return_energy:
        return T_cent, P, U_cent
    return T_cent, P


def mean_temperature(T_grid, P):
    """Probability-weighted mean grain temperature <T> = sum_i P_i T_i [K]."""
    return float(np.sum(P * T_grid))


def effective_radius_loss_rate(material, a_cm, T_grid, P):
    """Stochastic-heating-weighted grain radius shrink rate [cm s^-1].

    Computes the expectation value of ``radius_loss_rate`` over the
    temperature probability distribution ``P(T)``:

        <|da/dt|> = sum_i  P_i * |da/dt|(T_i)

    Because sublimation is exponentially sensitive to temperature, the
    stochastic average can exceed the single-equilibrium-temperature value
    by many orders of magnitude for small grains whose temperature spikes
    far above their time-averaged value.

    Parameters
    ----------
    material : str
        ``'graphite'`` or ``'silicate'``.
    a_cm : float
        Grain radius in cm.
    T_grid : np.ndarray
        Temperature bin centres [K], as returned by
        ``adaptive_temperature_distribution`` or ``temperature_distribution``.
    P : np.ndarray
        Corresponding normalised probabilities (same length as ``T_grid``).

    Returns
    -------
    float
        Probability-weighted radius loss rate ``<|da/dt|>`` in cm s^-1.
    """
    rates = np.array([radius_loss_rate(material, a_cm, float(Ti)) for Ti in T_grid])
    return float(np.dot(P, rates))


def effective_sublimation_timescale(material, a_cm, T_grid, P):
    """Stochastic sublimation timescale tau = a / <|da/dt|> in seconds.

    Following GD89, the lifetime against sublimation is tau = a / <da/dt>,
    where <da/dt> is the expectation value of the radius contraction rate
    over the stochastic temperature distribution P(T).

    Returns ``np.inf`` when the effective rate is zero (grain too cold to
    sublimate at any temperature in the distribution).

    Parameters
    ----------
    material : str
        ``'graphite'`` or ``'silicate'``.
    a_cm : float
        Grain radius in cm.
    T_grid, P : np.ndarray
        Temperature grid [K] and normalised probability distribution, as
        returned by ``adaptive_temperature_distribution``.

    Returns
    -------
    float
        Stochastic sublimation timescale in seconds.
    """
    da_dt_mean = effective_radius_loss_rate(material, a_cm, T_grid, P)
    return float(a_cm / da_dt_mean) if da_dt_mean > 0.0 else np.inf


def _significant_temperature_range(T_grid, P, rel_threshold=1e-20):
    """Temperature range where P(T)/Pmax exceeds ``rel_threshold``.

    Implements the range estimate used by the SKIRT heuristic (Camps et al.
    2015): the interval ``[Tmin, Tmax]`` spanned by the bins whose
    probability is above ``1e-20`` times the peak probability.

    Returns
    -------
    (float, float)
        ``(Tmin, Tmax)``. If no probability is positive, returns the grid
        endpoints.
    """
    Pmax = P.max() if P.size else 0.0
    if Pmax <= 0.0:
        return float(T_grid[0]), float(T_grid[-1])
    mask = P > rel_threshold * Pmax
    T_sel = T_grid[mask]
    return float(T_sel.min()), float(T_sel.max())


def adaptive_temperature_distribution(material, a_cm, wav_xs_cm, C_abs,
                                      wav_field, field, wav_em=None,
                                      n_bins_coarse=40, n_bins_wide=200,
                                      n_bins_narrow=300, T_min=2.0,
                                      delta_T_equilibrium=10.0,
                                      delta_T_split=200.0):
    """Grain temperature distribution via the SKIRT adaptive heuristic.

    Implements the bin-selection procedure of Camps et al. (2015, A&A 580,
    A87; section 4.3), which chooses between an explicit stochastic-heating
    calculation and the equilibrium (delta-function) limit, and refines the
    energy grid to the temperature range that actually carries probability:

    1. compute the equilibrium temperature ``Teq``;
    2. use a coarse grid (grid A) to estimate the range ``[Tmin, Tmax]``
       where ``P(T)/Pmax > 1e-20`` and ``deltaT = Tmax - Tmin``;
    3. if ``deltaT < 10 K`` or ``Tmax < Teq`` -> treat as equilibrium at
       ``Teq`` and return;
    4. recompute ``P(T)`` on a grid restricted to ``[Tmin, Tmax]`` using a
       wide grid (grid B, ``deltaT > 200 K``) or a narrow grid (grid C,
       ``deltaT < 200 K``);
    5. update the significant range from the refined calculation;
    6. re-apply the equilibrium test (``deltaT < 10 K`` or ``Tmax < Teq``);
    7. return ``P(T)`` over the final significant range.

    Parameters
    ----------
    material, a_cm, wav_xs_cm, C_abs, wav_field, field, wav_em
        As in :func:`temperature_distribution`.
    n_bins_coarse, n_bins_wide, n_bins_narrow : int
        Bin counts for grids A, B and C respectively.
    T_min : float
        Temperature floor of the energy grid in K.
    delta_T_equilibrium : float
        Width threshold (K) below which the grain is treated as being at
        radiative equilibrium (avoids the delta-function instability).
    delta_T_split : float
        Width threshold (K) separating the wide (B) and narrow (C) grids.

    Returns
    -------
    dict
        ``{'T': T_grid, 'P': P, 'T_eq': Teq, 'is_equilibrium': bool,
        'delta_T': deltaT, 'T_range': (Tmin, Tmax)}``. When
        ``is_equilibrium`` is ``True`` the distribution is a single bin at
        ``Teq`` with probability 1.
    """
    if wav_em is None:
        wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    # Step 1: equilibrium temperature.
    T_eq = compute_bin_temperature(a_cm, wav_xs_cm, C_abs, wav_field, field, wav_em=wav_em)

    def _equilibrium_result(delta_T, T_range):
        return {
            'T': np.array([T_eq]),
            'P': np.array([1.0]),
            'T_eq': T_eq,
            'is_equilibrium': True,
            'delta_T': delta_T,
            'T_range': T_range,
        }

    # Step 2: coarse grid A over the auto-estimated (equilibrium + photons) range.
    T_A, P_A = temperature_distribution(material, a_cm, wav_xs_cm, C_abs,
                                        wav_field, field, n_bins=n_bins_coarse,
                                        T_min=T_min, T_max=None, wav_em=wav_em)
    Tmin1, Tmax1 = _significant_temperature_range(T_A, P_A)
    delta_T1 = Tmax1 - Tmin1

    # Step 3: equilibrium shortcut.
    if delta_T1 < delta_T_equilibrium or Tmax1 < T_eq:
        return _equilibrium_result(delta_T1, (Tmin1, Tmax1))

    # Step 4: refine on grid B (wide) or grid C (narrow), restricted to the range.
    n_fine = n_bins_wide if delta_T1 > delta_T_split else n_bins_narrow
    T_min_fine = max(T_min, Tmin1)
    T_f, P_f = temperature_distribution(material, a_cm, wav_xs_cm, C_abs,
                                        wav_field, field, n_bins=n_fine,
                                        T_min=T_min_fine, T_max=Tmax1, wav_em=wav_em)

    # Step 5: update the significant range.
    Tmin2, Tmax2 = _significant_temperature_range(T_f, P_f)
    delta_T2 = Tmax2 - Tmin2

    # Step 6: re-apply the equilibrium test on the refined distribution.
    if delta_T2 < delta_T_equilibrium or Tmax2 < T_eq:
        return _equilibrium_result(delta_T2, (Tmin2, Tmax2))

    # Step 7: restrict to the final significant range and renormalise.
    mask = (T_f >= Tmin2) & (T_f <= Tmax2)
    T_out = T_f[mask]
    P_out = P_f[mask]
    total = P_out.sum()
    if total > 0.0:
        P_out = P_out / total
    return {
        'T': T_out,
        'P': P_out,
        'T_eq': T_eq,
        'is_equilibrium': False,
        'delta_T': delta_T2,
        'T_range': (Tmin2, Tmax2),
    }


def _temperature_distribution_result(material, a_cm, wav_xs_cm, C_abs,
                                     wav_field, field, wav_em=None,
                                     method='full',
                                     n_bins_full=300,
                                     n_bins_coarse=40,
                                     n_bins_wide=200,
                                     n_bins_narrow=300,
                                     T_min=2.0):
    """Return temperature distribution using either full GD89 or SKIRT adaptive.

    Parameters mirror ``temperature_distribution`` and
    ``adaptive_temperature_distribution``. ``method`` accepts:

    - ``'full'``: always solve the full GD89 transition-matrix distribution.
    - ``'adaptive'``: use the SKIRT heuristic (Camps et al. 2015).
    """
    method = str(method).lower()
    if wav_em is None:
        wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    if method == 'adaptive':
        return adaptive_temperature_distribution(
            material, a_cm, wav_xs_cm, C_abs, wav_field, field,
            wav_em=wav_em,
            n_bins_coarse=n_bins_coarse,
            n_bins_wide=n_bins_wide,
            n_bins_narrow=n_bins_narrow,
            T_min=T_min,
        )
    if method == 'full':
        T_eq = compute_bin_temperature(a_cm, wav_xs_cm, C_abs, wav_field, field, wav_em=wav_em)
        T_grid, P = temperature_distribution(
            material, a_cm, wav_xs_cm, C_abs, wav_field, field,
            n_bins=n_bins_full,
            T_min=T_min,
            T_max=None,
            wav_em=wav_em,
        )
        Tmin, Tmax = _significant_temperature_range(T_grid, P)
        return {
            'T': T_grid,
            'P': P,
            'T_eq': T_eq,
            'is_equilibrium': False,
            'delta_T': Tmax - Tmin,
            'T_range': (Tmin, Tmax),
        }
    raise ValueError("method must be 'full' or 'adaptive'.")


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------
def plot_sublimation(config_path=None,
                     G0_min=1.0, G0_max=1e12, n_G0=60,
                     dist_min_pc=3e-4, dist_max_pc=1.0, n_dist=60,
                     star_radius_rsun=10.0,
                     output_dir=None, filename='dust_sublimation.pdf',
                     quantity='timescale',
                     temperature_method='adaptive',
                     n_bins_full=300,
                     n_bins_coarse=40,
                     n_bins_fine=300,
                     T_min=2.0,
                     show=False):
    """Plot GD89 dust sublimation for each DustBin in the JSON config.

    Two panels share the y-axis:

    - Left: Mathis ISRF heating, sublimation timescale versus ``G0``.
    - Right: O6V stellar heating, sublimation timescale versus distance
      from the star.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the grain-size JSON configuration. Defaults to the active
        configuration.
    G0_min, G0_max, n_G0 : float, float, int
        Range and sampling of the Mathis ISRF scaling factor.
    dist_min_pc, dist_max_pc, n_dist : float, float, int
        Range and sampling of the distance from the O6V star, in parsec.
    star_radius_rsun : float
        Radius of the O6V star in solar radii.
    output_dir : str, optional
        Directory where the figure is saved. Defaults to the current dir.
    filename : str
        Output file name.
    quantity : str
        'timescale' to plot tau_sub [yr] (default) or 'rate' to plot the
        radius loss rate |da/dt| [cm s^-1].
    """
    quantity = str(quantity).lower()
    if quantity not in ('timescale', 'rate'):
        raise ValueError("quantity must be 'timescale' or 'rate'.")

    dust_bins = _load_dust_bins(config_path)
    if not dust_bins:
        raise ValueError('No DustBin entries found in the configuration.')
    prepared = _prepare_bin_cross_sections(dust_bins)

    G0_list = np.logspace(np.log10(G0_min), np.log10(G0_max), n_G0)
    dist_list = np.logspace(np.log10(dist_min_pc), np.log10(dist_max_pc), n_dist)

    # Reusable emission grid.
    wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(prepared)))
    linestyles = {'graphite': '-', 'silicate': '--'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharey=True,
                             facecolor='w', edgecolor='k')

    for bin_data, color in zip(prepared, colors):
        material = bin_data['material']
        a0_cm = bin_data['a0_cm']
        wav_xs = bin_data['wav_cm']
        C_abs = bin_data['C_abs']
        ls = linestyles.get(material, '-')
        label = f"{bin_data['id']} ({material}, " \
                f"$a_0$={a0_cm * 1e4:.3g} $\\mu$m)"

        # --- Mathis ISRF panel ---
        y_mathis = np.zeros(len(G0_list))
        for i, G0 in enumerate(G0_list):
            wav_f, field = mathis_radiation_field_spectrum(G0=G0)
            res = _temperature_distribution_result(
                material, a0_cm, wav_xs, C_abs, wav_f, field,
                wav_em=wav_em,
                method=temperature_method,
                n_bins_full=n_bins_full,
                n_bins_coarse=n_bins_coarse,
                n_bins_wide=n_bins_fine,
                n_bins_narrow=n_bins_fine,
                T_min=T_min,
            )
            T_grid, P = res['T'], res['P']
            if quantity == 'timescale':
                y_mathis[i] = effective_sublimation_timescale(material, a0_cm, T_grid, P) / _YR
            else:
                y_mathis[i] = effective_radius_loss_rate(material, a0_cm, T_grid, P)
        # Non-finite (cold-grain) values are masked so the log axis can autoscale.
        axes[0].plot(G0_list, np.where(np.isfinite(y_mathis), y_mathis, np.nan),
                     color=color, linestyle=ls, linewidth=2.0, label=label)

        # --- O6V star panel ---
        y_star = np.zeros(len(dist_list))
        for i, dpc in enumerate(dist_list):
            wav_f, field = o6v_radiation_field_spectrum(dpc, star_radius_rsun=star_radius_rsun)
            res = _temperature_distribution_result(
                material, a0_cm, wav_xs, C_abs, wav_f, field,
                wav_em=wav_em,
                method=temperature_method,
                n_bins_full=n_bins_full,
                n_bins_coarse=n_bins_coarse,
                n_bins_wide=n_bins_fine,
                n_bins_narrow=n_bins_fine,
                T_min=T_min,
            )
            T_grid, P = res['T'], res['P']
            if quantity == 'timescale':
                y_star[i] = effective_sublimation_timescale(material, a0_cm, T_grid, P) / _YR
            else:
                y_star[i] = effective_radius_loss_rate(material, a0_cm, T_grid, P)
        axes[1].plot(dist_list, np.where(np.isfinite(y_star), y_star, np.nan),
                     color=color, linestyle=ls, linewidth=2.0, label=label)

    if quantity == 'timescale':
        ylabel = r'$\tau_{\rm sub}$ [yr]'
        for ax in axes:
            ax.set_ylim(1e-2, 1e25)
            # Reference line at the age of the Universe.
            ax.axhline(1.38e10, color='grey', linestyle=':', linewidth=1.2)
        axes[0].text(0.04, 0.06, 'age of Universe', color='grey', fontsize=10,
                     transform=axes[0].transAxes)
    else:
        ylabel = r'$|da/dt|$ [cm s$^{-1}$]'

    axes[0].set_xlabel(r'$G_0$ (Mathis ISRF)', fontsize=15)
    axes[0].set_ylabel(ylabel, fontsize=15)
    axes[1].set_xlabel(r'Distance from O6V star [pc]', fontsize=15)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both', axis='both', direction='in', labelsize=12)

    axes[0].legend(loc='best', frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.97, bottom=0.13, left=0.09, right=0.98, wspace=0.05)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, format=os.path.splitext(filename)[1].lstrip('.') or 'pdf', dpi=300)
        print('Saved dust sublimation plot to', out_path)
    else:
        out_path = None
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_temperature_distribution(config_path=None,
                                  G0_values=(1e2, 1e4, 1e6),
                                  n_bins=128, T_min=2.0,
                                  temperature_method='adaptive',
                                  output_dir=None,
                                  filename='dust_temperature_distribution.pdf',
                                  show=False):
    """Plot the stochastic grain temperature distribution per DustBin.

    One panel per value in ``G0_values`` (Mathis ISRF scaling). Each panel
    shows the probability distribution ``P(T)`` for every non-PAH DustBin,
    with a vertical dotted line at the corresponding radiative-equilibrium
    temperature for reference. Small grains show broad, skewed distributions
    (stochastic heating), while large grains pile up at their equilibrium
    temperature.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the grain-size JSON configuration.
    G0_values : sequence of float
        Mathis ISRF scaling factors, one sub-panel each.
    n_bins : int
        Number of internal-energy bins for the transition matrix.
    T_min : float
        Lower temperature bound of the energy grid in K.
    output_dir : str, optional
        Directory where the figure is saved. Defaults to the current dir.
    filename : str
        Output file name.
    """
    dust_bins = _load_dust_bins(config_path)
    if not dust_bins:
        raise ValueError('No DustBin entries found in the configuration.')
    prepared = _prepare_bin_cross_sections(dust_bins)

    wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(prepared)))
    linestyles = {'graphite': '-', 'silicate': '--'}

    G0_values = list(G0_values)
    n_panels = len(G0_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.4), dpi=300,
                             sharey=True, facecolor='w', edgecolor='k')
    axes = np.atleast_1d(axes)

    # Track the peak dP/dT so the equilibrium delta-function markers can be
    # placed at a representative height rather than spanning the whole axis.
    peak_density = {id(ax): 1e-3 for ax in axes}

    for ax, G0 in zip(axes, G0_values):
        wav_f, field = mathis_radiation_field_spectrum(G0=G0)
        for bin_data, color in zip(prepared, colors):
            material = bin_data['material']
            a0_cm = bin_data['a0_cm']
            wav_xs = bin_data['wav_cm']
            C_abs = bin_data['C_abs']
            ls = linestyles.get(material, '-')
            label = f"{bin_data['id']} ({material}, " \
                    f"$a_0$={a0_cm * 1e4:.3g} $\\mu$m)"

            res = _temperature_distribution_result(
                material, a0_cm, wav_xs, C_abs, wav_f, field,
                wav_em=wav_em,
                method=temperature_method,
                n_bins_full=n_bins,
                n_bins_coarse=max(10, n_bins // 3),
                n_bins_wide=n_bins,
                n_bins_narrow=n_bins,
                T_min=T_min,
            )
            T_grid, P, T_eq = res['T'], res['P'], res['T_eq']

            if res['is_equilibrium']:
                # Equilibrium grain: the temperature distribution is a delta
                # function at Teq (the stochastic method is intentionally not
                # used here, per the Camps et al. 2015 deltaT < 10 K rule).
                # Draw it as a clearly-marked stem so it reads as a sharp peak
                # rather than a gridline.
                ax.plot([T_eq], [1.0], marker='v', markersize=8, color=color,
                        linestyle='none',
                        label=label + r' (eq. $\delta$)', zorder=5,
                        clip_on=False, transform=ax.get_xaxis_transform())
                ax.axvline(T_eq, color=color, linestyle=ls, linewidth=1.5,
                           alpha=0.8, zorder=4)
            else:
                # Convert the binned probabilities to a probability density
                # dP/dT so distributions on the same axis are comparable.
                dT = np.gradient(T_grid)
                dP_dT = np.zeros_like(P)
                good = dT > 0
                dP_dT[good] = P[good] / dT[good]
                ax.plot(T_grid, dP_dT, color=color, linestyle=ls, linewidth=2.0, label=label)
                if dP_dT.size:
                    peak_density[id(ax)] = max(peak_density[id(ax)], dP_dT.max())

        ax.set_title(rf'$G_0 = 10^{{{int(np.log10(G0))}}}$', fontsize=13)
        ax.set_xlabel(r'$T$ [K]', fontsize=15)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both', axis='both', direction='in', labelsize=12)

    axes[0].set_ylabel(r'$dP/dT$ [K$^{-1}$]', fontsize=15)
    axes[0].legend(loc='best', frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.93, bottom=0.13, left=0.08, right=0.98, wspace=0.05)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, format=os.path.splitext(filename)[1].lstrip('.') or 'pdf', dpi=300)
        print('Saved dust temperature distribution plot to', out_path)
    else:
        out_path = None
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


if __name__ == '__main__':
    plot_sublimation()


def write_sublimation_rate_tables(config_path=None,
                                  T_min=200.0, T_max=4000.0, n_T=300,
                                  output_dir=None):
    """Write Fortran-readable sublimation-rate tables for every non-PAH DustBin.

    One file per DustBin is written to ``output_dir`` (default:
    ``model_data/dust_sublimation/``).  Each file contains two columns:

        column 1 : dust temperature  T_d  [K]
        column 2 : fractional sublimation rate  epsilon = |da/dt| / a  [s^-1]

    The temperature grid is log-spaced between ``T_min`` and ``T_max``.
    To avoid interpolating very small / negligible values of the sublimation rate,
    we set epsilon to exactly 0.0 for dust temperatures where the sublimation
    timescale is longer than 10 times the age of the Universe (10 * 1.38e10 yr).

    Parameters
    ----------
    config_path : str or Path, optional
        Grain-size JSON configuration. Defaults to the active configuration.
    T_min, T_max : float
        Temperature range in K.
    n_T : int
        Number of log-spaced temperature points.
    output_dir : str, optional
        Output directory.  Defaults to ``model_data/dust_sublimation``.

    Returns
    -------
    list of str
        Paths of the files that were written.
    """
    import datetime

    dust_bins = _load_dust_bins(config_path)
    if not dust_bins:
        raise ValueError('No DustBin entries found in the configuration.')
    prepared = _prepare_bin_cross_sections(dust_bins)

    if output_dir is None:
        output_dir = os.path.join(
            str(get_repo_root()), 'model_data', 'dust_sublimation')
    os.makedirs(output_dir, exist_ok=True)

    T_grid = np.logspace(np.log10(T_min), np.log10(T_max), n_T)  # [K]
    date_str = datetime.date.today().isoformat()
    written = []

    # Limit timescale threshold: 10x age of Universe in seconds
    age_universe_yr = 1.38e10
    limit_timescale_s = 10.0 * age_universe_yr * _YR

    for bin_data in prepared:
        material  = bin_data['material']
        a0_cm     = bin_data['a0_cm']
        bin_id    = bin_data['id']

        dadt = np.array([radius_loss_rate(material, a0_cm, float(T))
                         for T in T_grid])                  # [cm s^-1]
        
        # Calculate sublimation timescale against each temperature
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_sub = np.where(dadt > 0.0, a0_cm / dadt, np.inf)

        # Filter: keep only where dadt > 0.0 and tau_sub <= limit_timescale_s
        mask = (dadt > 0.0) & (tau_sub <= limit_timescale_s)
        T_filtered = T_grid[mask]
        epsilon_filtered = (dadt / a0_cm)[mask]

        from pycalima.models.grain_size_config import get_header_lines
        headers = get_header_lines(
            title="CALIMA dust sublimation rate table",
            script_name="models/dust_radiation/dust_sublimation.py",
            bin_info=f"Dust bin: {bin_id}, Material: {material}, Grain radius (representative): {a0_cm:.6e} cm ({a0_cm * 1e4:.6e} micron)",
            val_desc="Columns: T_d [K] (dust temperature) | log10(epsilon/[s^-1]) (fractional sublimation rate)"
        )
        fname = os.path.join(output_dir, f'sublimation_rate_{bin_id}.dat')
        with open(fname, 'w') as fh:
            for line in headers:
                fh.write(f"{line}\n")
            fh.write(
                f'# Method    : GD89 thermal sublimation (Guhathakurta & Draine\n'
                f'#             1989, ApJ 345, 230) with microcanonical correction\n'
                f'#             factor (eq. 3.11 + Gamma-function ratio, eq. 3.16).\n'
                f'#             Binding energy B(N) = B_inf - sigma*N^(-1/3),\n'
                f'#             B_inf/k_B = {GD89_PARAMS[material]["B_saturation_temperature_K"]:.1f} K,\n'
                f'#             sigma/k_B = {GD89_PARAMS[material]["surface_free_energy_K"]:.1f} K.\n'
                f'# Sublimation rate: epsilon = |da/dt| / a  [s^-1]\n'
                f'#   where |da/dt| = (V_mon / (4 pi a^2)) * dN/dt [cm s^-1]\n'
                f'#   and   dN/dt   = 4 pi a^2 * J(T) * f_corr      [s^-1]\n'
                f'# Only dust temperatures where the sublimation timescale is\n'
                f'# less than or equal to 10x the age of the Universe are saved\n'
                f'# (preventing tiny value interpolations / micro-level noise).\n'
                f'# N rows : {len(T_filtered)}\n'
                f'#\n'
            )
            # Two fixed-width columns: temperature (E14.6) and rate (E14.6).
            # Fortran can read these with:
            #   READ(unit, '(E14.6, 1X, E14.6)') T_d, epsilon
            for T, eps in zip(T_filtered, epsilon_filtered):
                fh.write(f'{T:14.6E} {np.log10(eps):14.6E}\n')

        print(f'Written {fname}')
        written.append(fname)

    return written


def export_dust_sublimation(config_path=None, output_dir=None):
    """Export all dust sublimation tables and figures to the output directory.

    This compiles:
      1. Sputtering/sublimation rate tables for non-PAH DustBins.
      2. Plot of sublimation rate vs dust temperature.
      3. Plot of G0 and O6V sublimation timescales.

    Parameters
    ----------
    config_path : str or Path, optional
        Grain-size JSON configuration. Defaults to the active configuration.
    output_dir : str, optional
        Output directory. Defaults to ``model_data/dust_sublimation``.

    Returns
    -------
    dict
        Metadata dictionary summarizing files written.
    """
    if output_dir is None:
        output_dir = os.path.join(
            str(get_repo_root()), 'model_data', 'dust_sublimation')
    os.makedirs(output_dir, exist_ok=True)

    tables = write_sublimation_rate_tables(
        config_path=config_path, output_dir=output_dir)

    rate_plot = plot_sublimation_rate_vs_temperature(
        config_path=config_path, output_dir=output_dir,
        filename='sublimation_rate_vs_T.pdf', show=False)

    timescale_plot = plot_sublimation(
        config_path=config_path, output_dir=output_dir,
        filename='dust_sublimation.pdf', show=False)

    return {
        'status': 'Success',
        'output_dir': output_dir,
        'tables': tables,
        'plots': [rate_plot, timescale_plot]
    }


def write_erosion_rate_tables(*args, **kwargs):
    """Old deprecated name of write_sublimation_rate_tables."""
    import warnings
    warnings.warn(
        "write_erosion_rate_tables is deprecated, use "
        "write_sublimation_rate_tables instead.",
        DeprecationWarning, stacklevel=2
    )
    return write_sublimation_rate_tables(*args, **kwargs)

    return written


def plot_sublimation_rate_vs_temperature(config_path=None,
                                         T_min=200.0, T_max=4000.0, n_T=300,
                                         output_dir=None,
                                         filename='sublimation_rate_vs_T.pdf',
                                         show=False):
    """Grain erosion rate as a function of dust temperature.

    Single panel: erosion rate ``epsilon = |da/dt| / a`` [s^-1] vs ``T`` [K]
    for every non-PAH DustBin in the JSON configuration.

    ``epsilon`` is the fractional radius loss rate (equivalently
    ``1 / tau_sub``): it directly answers "per second, what fraction of the
    grain radius is eroded?".  Reference horizontal lines mark
    astrophysically relevant erosion rates corresponding to timescales of
    1 yr, 10^3 yr, 10^6 yr, and the age of the Universe.

    Parameters
    ----------
    config_path : str or Path, optional
        Grain-size JSON configuration. Defaults to the active configuration.
    T_min, T_max : float
        Temperature range in K.
    n_T : int
        Number of temperature grid points (log-spaced).
    output_dir : str, optional
        Output directory. Defaults to the current working directory.
    filename : str
        Output file name.
    """
    dust_bins = _load_dust_bins(config_path)
    if not dust_bins:
        raise ValueError('No DustBin entries found in the configuration.')
    prepared = _prepare_bin_cross_sections(dust_bins)

    T_grid = np.logspace(np.log10(T_min), np.log10(T_max), n_T)  # [K]

    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(prepared)))
    linestyles = {'graphite': '-', 'silicate': '--'}

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.6), dpi=300,
                           facecolor='w', edgecolor='k')

    for idx_bin, (bin_data, color) in enumerate(zip(prepared, colors)):
        material = bin_data['material']
        a0_cm = bin_data['a0_cm']
        ls = linestyles.get(material, '-')
        label = f"{bin_data['id']} ({material}, " \
                f"$a_0$={a0_cm * 1e4:.3g} $\\mu$m)"

        dadt = np.array([radius_loss_rate(material, a0_cm, float(T))
                         for T in T_grid])              # [cm s^-1]
        epsilon = np.where(dadt > 0.0, dadt / a0_cm, np.nan)  # [s^-1]

        ax.plot(T_grid, epsilon, color=color, linestyle=ls,
                linewidth=2.0, label=label)

        # Waxman & Draine (2000) reference formula for the same grain size:
        a_5 = a0_cm / 1.0e-5
        if material == 'silicate':
            tau_wd = 6.36e3 * a_5 * np.exp(
                np.clip(6.81e4 * (1.0 / T_grid - 1.0 / 1800.0), -700.0, 700.0))
        else:  # graphite
            tau_wd = 1.36 * a_5 * np.exp(
                np.clip(8.12e4 * (1.0 / T_grid - 1.0 / 3000.0), -700.0, 700.0))
        eps_wd = np.where(tau_wd > 0.0, 1.0 / tau_wd, np.nan)  # [s^-1]
        # Label only the first graphite/silicate entry as WD00 reference to avoid legend clutter
        wd_label = f'WD2000 ({material})' if (idx_bin == 0 or idx_bin == 2) else None
        ax.plot(T_grid, eps_wd, color=color, linestyle=':', linewidth=1.5,
                alpha=0.8, label=wd_label)

    # Reference lines at erosion rates corresponding to round timescales.
    ref_timescales = [
        (1.0,    '1 yr'),
        (1e3,    r'$10^3$ yr'),
        (1e6,    r'$10^6$ yr'),
        (1.38e10, 'age of Universe'),
    ]
    for t_yr, t_label in ref_timescales:
        eps_ref = 1.0 / (t_yr * _YR)          # [s^-1]
        ax.axhline(eps_ref, color='grey', linestyle=':', linewidth=1.0)
        ax.text(T_min * 1.05, eps_ref * 1.5, t_label,
                color='grey', fontsize=8, va='bottom')

    ax.set_xlabel(r'$T_{\rm d}$ [K]', fontsize=14)
    ax.set_ylabel(r'$\epsilon = |da/dt|\,/\,a$ [s$^{-1}$]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.set_ylim([1e-20,1e10])
    ax.tick_params(which='both', axis='both', direction='in', labelsize=12)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.97, bottom=0.13, left=0.13, right=0.97)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, format=os.path.splitext(filename)[1].lstrip('.') or 'pdf', dpi=300)
        print('Saved sublimation rate vs temperature plot to', out_path)
    else:
        out_path = None
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Grain lifetime vs number of atoms
# ---------------------------------------------------------------------------

def plot_grain_lifetime_vs_N(
        N_min=1e1, N_max=1e12, n_N=200,
        G0_values=(1e0, 1e2, 1e4, 1e6),
        materials=('graphite', 'silicate'),
    temperature_method='adaptive',
    n_bins_full=300,
    n_bins_coarse=40,
    n_bins_fine=300,
    T_min=2.0,
        xscale='log', yscale='log',
        output_dir=None, filename='grain_lifetime_vs_N.pdf',
        show=False):
    """Grain sublimation lifetime vs number of monomers for the Mathis ISRF.

    For each material, the grain radius is derived directly from N using the
    GD89 bulk density and monomer mass:

        a = ( 3 N m_mon / (4 pi rho) )^{1/3}

    Absorption cross sections are obtained from the Draine optical tables
    (interpolated to the exact grain size with
    ``interpolate_cross_sections``).  The equilibrium grain temperature is
    solved self-consistently from the absorbed/emitted power balance, and
    the stochastic temperature distribution (Camps et al. 2015) is used for
    grains that are far from equilibrium (small N / weak field).

    Parameters
    ----------
    N_min, N_max : float
        Range of monomer counts (number of C or MgFeSiO4 formula units).
    n_N : int
        Number of log-spaced N points.
    G0_values : sequence of float
        Mathis ISRF scaling factors; one curve per value per material.
    materials : sequence of str
        Subset of ``('graphite', 'silicate')`` to include.
    xscale : str
        Scale for the N axis: ``'log'`` (default) or ``'linear'``.
    yscale : str
        Scale for the lifetime axis: ``'log'`` (default) or ``'linear'``.
    output_dir : str, optional
        Directory where the figure is saved. If ``None`` the figure is not
        saved.
    filename : str
        Output file name.
    show : bool
        If ``True``, call ``plt.show()``; otherwise close the figure.
    """
    if xscale == 'linear':
        N_grid = np.linspace(N_min, N_max, n_N)
    else:
        N_grid = np.logspace(np.log10(N_min), np.log10(N_max), n_N)

    # Reusable emission wavelength grid
    wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4  # cm

    mat_colors = {'graphite': plt.cm.Blues, 'silicate': plt.cm.Oranges}
    mat_ls     = {'graphite': '-',          'silicate': '--'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300,
                           facecolor='w', edgecolor='k')

    for material in materials:
        params   = GD89_PARAMS[material]
        m_mon    = params['mass_amu'] * _AMU   # [g]
        rho      = params['rho']               # [g cm^-3]
        cmap     = mat_colors[material]
        ls       = mat_ls[material]
        g0_colors = cmap(np.linspace(0.35, 0.95, len(G0_values)))

        # Pre-load the Draine table once for this material so every
        # interpolate_cross_sections call can reuse it.
        if material == 'graphite':
            from pycalima.models.dust_radiation.dust_emission import dust_efficiencies
            import os as _os
            from pycalima.models.grain_size_config import get_repo_root as _get_repo
            _PATH_OPTICS = _os.path.join(str(_get_repo()), 'optical_props')
            _fname = _os.path.join(_PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
        else:
            from pycalima.models.dust_radiation.dust_emission import dust_efficiencies
            import os as _os
            from pycalima.models.grain_size_config import get_repo_root as _get_repo
            _PATH_OPTICS = _os.path.join(str(_get_repo()), 'optical_props')
            _fname = _os.path.join(_PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
        _data_table = dust_efficiencies(_fname)

        for G0, color in zip(G0_values, g0_colors):
            wav_f, field = mathis_radiation_field_spectrum(G0=G0)
            tau_arr = np.full(n_N, np.nan)

            for k, N in enumerate(N_grid):
                # Grain radius from number of monomers
                a_cm = (3.0 * N * m_mon / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
                a_um = a_cm * 1.0e4  # micron for interpolate_cross_sections

                # Absorption cross section at this exact size (Draine tables)
                _, wav_xs, _, C_abs, _ = interpolate_cross_sections(
                    material, a_um, data_table=_data_table)

                # Temperature distribution using either full GD89 matrix
                # or SKIRT adaptive heuristic.
                res = _temperature_distribution_result(
                    material, a_cm, wav_xs, C_abs, wav_f, field,
                    wav_em=wav_em,
                    method=temperature_method,
                    n_bins_full=n_bins_full,
                    n_bins_coarse=n_bins_coarse,
                    n_bins_wide=n_bins_fine,
                    n_bins_narrow=n_bins_fine,
                    T_min=T_min,
                )
                T_grid_grain, P = res['T'], res['P']

                tau_s = effective_sublimation_timescale(
                    material, a_cm, T_grid_grain, P)  # [s]

                if np.isfinite(tau_s) and tau_s > 0.0:
                    tau_arr[k] = tau_s

            exp_G0 = int(round(np.log10(G0)))
            label = fr'{material}, $G_0=10^{{{exp_G0}}}$'
            ax.plot(N_grid, tau_arr, color=color, linestyle=ls,
                    linewidth=1.8, label=label)

    # Reference timescales
    ref_lines = [
        (1.0,     '1 yr'),
        (1e3,     r'$10^3$ yr'),
        (1e6,     r'$10^6$ yr'),
        (1.38e10, 'age of Universe'),
    ]
    for t_yr, t_label in ref_lines:
        t_s = t_yr * _YR
        ax.axhline(t_s, color='grey', linestyle=':', linewidth=1.0)
        ax.text(N_min * 1.3, t_s * 1.6, t_label,
                color='grey', fontsize=8, va='bottom')

    # Secondary x-axis: grain radius for graphite (reference material)
    m_gra = GD89_PARAMS['graphite']['mass_amu'] * _AMU
    rho_gra = GD89_PARAMS['graphite']['rho']

    def N_to_a_um(N):
        return (3.0 * N * m_gra / (4.0 * np.pi * rho_gra)) ** (1.0 / 3.0) * 1e4

    def a_um_to_N(a):
        a_cm = a * 1e-4
        return 4.0 * np.pi * rho_gra * a_cm ** 3 / (3.0 * m_gra)

    ax2 = ax.secondary_xaxis('top', functions=(N_to_a_um, a_um_to_N))
    ax2.set_xlabel(r'$a$ [µm]  (graphite equivalent)', fontsize=11)
    ax2.tick_params(which='both', direction='in', labelsize=10)

    ax.set_xlabel(r'Number of monomers $N$', fontsize=14)
    ax.set_ylabel(r'Sublimation lifetime $\tau_{\rm sub}$ [s]', fontsize=14)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylim(1e9, 1e17)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', labelsize=12)
    ax.legend(loc='upper left', frameon=False, fontsize=8, ncol=2)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.12, right=0.97)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, format=os.path.splitext(filename)[1].lstrip('.') or 'pdf',
                    dpi=300)
        print('Saved grain lifetime vs N plot to', out_path)
    else:
        out_path = None
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_graphite_lifetime_gd89_variants_comparison(
        G0=1.0,
    n_model_points=80,
    temperature_method='full',
    n_bins_full=300,
    n_bins_coarse=50,
    n_bins_fine=300,
        output_dir=None,
        filename='graphite_fig3_variants_vs_model.png',
        summary_filename='graphite_fig3_set_closeness.txt',
        use_gd89_isrf=False,
        show=False):
    """Compare current graphite lifetimes to GD89 digitized variant curves.

    Produces one combined plot with:

    - current CALIMA model (solid line),
    - GD89 ``nocorr`` points,
    - GD89 ``STcorr`` points,
    - GD89 ``FScorr`` points,
    - GD89 full-correction points.

    Also writes a text summary ranking which GD89 set is closest to the
    current model in log-space (median and mean absolute dex offsets).

    Parameters
    ----------
    G0 : float
        Mathis ISRF scaling factor used for the model curve.
    n_model_points : int
        Number of log-spaced N samples for the model line.
    n_bins_coarse : int
        Coarse-bin count for ``adaptive_temperature_distribution``.
    n_bins_fine : int
        Fine-bin count (used for both wide and narrow adaptive grids).
    output_dir : str, optional
        Directory where outputs are written. Defaults to
        ``results/`` under the repository root.
    filename : str
        Output plot filename.
    summary_filename : str
        Output text summary filename.
    use_gd89_isrf : bool
        If ``True``, use the exact digitized Mathis et al. (1983) ISRF from
        the GD89 paper (csv format) instead of modified MMP83.
    show : bool
        If ``True``, call ``plt.show()``; otherwise close the figure.

    Returns
    -------
    dict
        Paths and ranking summary:
        ``{'plot_path': str, 'summary_path': str, 'ranking': list}``.
    """
    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'results')
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'nocorr': np.loadtxt(
            os.path.join(_EXTERNAL_DATA_DIR, 'sublimation_time_graphite_GD89_nocorr.csv'),
            delimiter=','),
        'STcorr': np.loadtxt(
            os.path.join(_EXTERNAL_DATA_DIR, 'sublimation_time_graphite_GD89_STcorr.csv'),
            delimiter=','),
        'FScorr': np.loadtxt(
            os.path.join(_EXTERNAL_DATA_DIR, 'sublimation_time_graphite_GD89_FScorr.csv'),
            delimiter=','),
        'full': np.loadtxt(
            os.path.join(_EXTERNAL_DATA_DIR, 'sublimation_time_graphite_GD89.csv'),
            delimiter=','),
    }

    wav_f, field = mathis_radiation_field_spectrum(G0=G0, use_gd89_isrf=use_gd89_isrf)
    wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    def _model_tau_from_N(N):
        a_angstrom = (float(N) / 0.470422) ** (1.0 / 3.0)  # GD89: physical total atom count mapping (rho=2.24, mu=12)
        a_cm = a_angstrom * 1e-8
        _, wav_xs, _, C_abs, _ = interpolate_cross_sections('graphite', a_cm * 1e4)
        res = _temperature_distribution_result(
            'graphite', a_cm, wav_xs, C_abs, wav_f, field,
            wav_em=wav_em,
            method=temperature_method,
            n_bins_full=n_bins_full,
            n_bins_coarse=n_bins_coarse,
            n_bins_wide=n_bins_fine,
            n_bins_narrow=n_bins_fine,
        )
        return effective_sublimation_timescale('graphite', a_cm, res['T'], res['P'])

    all_N = np.concatenate([arr[:, 0] for arr in datasets.values()])
    N_grid = np.logspace(np.log10(all_N.min()), np.log10(all_N.max()), n_model_points)
    tau_grid = np.array([_model_tau_from_N(N) for N in N_grid])
    logN_grid = np.log10(N_grid)
    logtau_grid = np.log10(tau_grid)

    ranking = []
    for name, arr in datasets.items():
        N_ref = arr[:, 0]
        tau_ref = arr[:, 1]
        tau_model = 10 ** np.interp(np.log10(N_ref), logN_grid, logtau_grid)
        ratio = tau_model / tau_ref
        dex = np.log10(ratio)
        ranking.append({
            'set': name,
            'median_abs_dex': float(np.median(np.abs(dex))),
            'mean_abs_dex': float(np.mean(np.abs(dex))),
            'min_ratio': float(np.min(ratio)),
            'max_ratio': float(np.max(ratio)),
            'median_ratio': float(np.median(ratio)),
        })

    ranking.sort(key=lambda row: row['median_abs_dex'])

    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, 'w') as fh:
        fh.write('Set  median_abs_dex  mean_abs_dex  min_ratio  max_ratio  median_ratio\n')
        for row in ranking:
            fh.write(
                f"{row['set']:7s} "
                f"{row['median_abs_dex']:14.6f} "
                f"{row['mean_abs_dex']:13.6f} "
                f"{row['min_ratio']:10.3e} "
                f"{row['max_ratio']:10.3e} "
                f"{row['median_ratio']:12.3e}\n"
            )

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0), dpi=180,
                           facecolor='w', edgecolor='k')

    ax.plot(N_grid, tau_grid, color='k', linewidth=2.0,
            label=f'CALIMA model ($G_0={G0:g}$)')

    style = {
        'nocorr': {'marker': 'o', 'color': '#1f77b4', 'size': 24, 'label': 'GD89 nocorr'},
        'STcorr': {'marker': 's', 'color': '#ff7f0e', 'size': 26, 'label': 'GD89 STcorr only'},
        'FScorr': {'marker': '^', 'color': '#2ca02c', 'size': 28, 'label': 'GD89 FScorr only'},
        'full': {'marker': 'D', 'color': '#d62728', 'size': 28, 'label': 'GD89 full corrections'},
    }

    for name, arr in datasets.items():
        ax.scatter(arr[:, 0], arr[:, 1],
                   s=style[name]['size'],
                   marker=style[name]['marker'],
                   color=style[name]['color'],
                   label=style[name]['label'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of atoms $N$ (graphite)', fontsize=13)
    ax.set_ylabel(r'Lifetime $\tau_{\rm sub}$ [s]', fontsize=13)
    ax.set_title('Graphite lifetime: CALIMA vs GD89 Fig. 3 variants', fontsize=13)
    ax.grid(alpha=0.25, which='both')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis='both', direction='in', labelsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path,
                format=os.path.splitext(filename)[1].lstrip('.') or 'png',
                dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print('Saved plot:', plot_path)
    print('Saved summary:', summary_path)
    print('Closeness ranking (best first):')
    for row in ranking:
        print(
            f"  {row['set']:7s} "
            f"median|dex|={row['median_abs_dex']:.3f}, "
            f"mean|dex|={row['mean_abs_dex']:.3f}, "
            f"ratio=[{row['min_ratio']:.2e}, {row['max_ratio']:.2e}], "
            f"median ratio={row['median_ratio']:.2e}"
        )

    return {
        'plot_path': plot_path,
        'summary_path': summary_path,
        'ranking': ranking,
    }


def plot_silicate_lifetime_gd89_comparison(
        G0=1.0,
        n_model_points=80,
        temperature_method='full',
        n_bins_full=300,
        n_bins_coarse=50,
        n_bins_fine=300,
        output_dir=None,
        filename='silicate_lifetime_vs_gd89.png',
        summary_filename='silicate_set_closeness.txt',
        use_gd89_isrf=False,
        show=False):
    """Compare current silicate lifetimes to the available GD89 silicate line.

    Produces one combined plot with:

    - current CALIMA silicate model (solid line),
    - GD89 silicate digitized line (single available dataset).

    Writes a text summary with log-space closeness metrics between the
    model and GD89 silicate points.

    Returns
    -------
    dict
        ``{'plot_path': str, 'summary_path': str, 'metrics': dict}``.
    """
    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'results')
    os.makedirs(output_dir, exist_ok=True)

    gd89 = np.loadtxt(
        os.path.join(_EXTERNAL_DATA_DIR, 'sublimation_time_silicate_GD89.csv'),
        delimiter=',')

    wav_f, field = mathis_radiation_field_spectrum(G0=G0, use_gd89_isrf=use_gd89_isrf)
    wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    params = GD89_PARAMS['silicate']
    m_mon = params['mass_amu'] * _AMU
    rho = params['rho']

    def _model_tau_from_N(N):
        a_angstrom = (float(N) / 0.44145) ** (1.0 / 3.0)  # GD89: physical total atom count mapping (rho=3.5, mu=20.1)
        a_cm = a_angstrom * 1e-8
        _, wav_xs, _, C_abs, _ = interpolate_cross_sections('silicate', a_cm * 1e4)
        res = _temperature_distribution_result(
            'silicate', a_cm, wav_xs, C_abs, wav_f, field,
            wav_em=wav_em,
            method=temperature_method,
            n_bins_full=n_bins_full,
            n_bins_coarse=n_bins_coarse,
            n_bins_wide=n_bins_fine,
            n_bins_narrow=n_bins_fine,
        )
        return effective_sublimation_timescale('silicate', a_cm, res['T'], res['P'])

    N_ref = gd89[:, 0]
    tau_ref = gd89[:, 1]

    N_grid = np.logspace(np.log10(N_ref.min()), np.log10(N_ref.max()), n_model_points)
    tau_grid = np.array([_model_tau_from_N(N) for N in N_grid])
    tau_model_ref = 10 ** np.interp(np.log10(N_ref), np.log10(N_grid), np.log10(tau_grid))

    ratio = tau_model_ref / tau_ref
    dex = np.log10(ratio)
    metrics = {
        'set': 'silicate',
        'median_abs_dex': float(np.median(np.abs(dex))),
        'mean_abs_dex': float(np.mean(np.abs(dex))),
        'min_ratio': float(np.min(ratio)),
        'max_ratio': float(np.max(ratio)),
        'median_ratio': float(np.median(ratio)),
    }

    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, 'w') as fh:
        fh.write('Set  median_abs_dex  mean_abs_dex  min_ratio  max_ratio  median_ratio\n')
        fh.write(
            f"{metrics['set']:8s} "
            f"{metrics['median_abs_dex']:14.6f} "
            f"{metrics['mean_abs_dex']:13.6f} "
            f"{metrics['min_ratio']:10.3e} "
            f"{metrics['max_ratio']:10.3e} "
            f"{metrics['median_ratio']:12.3e}\n"
        )

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0), dpi=180,
                           facecolor='w', edgecolor='k')

    ax.plot(N_grid, tau_grid, color='k', linewidth=2.0,
            label=f'CALIMA silicate model ($G_0={G0:g}$)')
    ax.plot(N_ref, tau_ref, color='#d62728', linewidth=1.6,
            linestyle='--', label='GD89 silicate line')
    ax.scatter(N_ref, tau_ref, s=22, color='#d62728', marker='o')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of atoms $N$ (silicate)', fontsize=13)
    ax.set_ylabel(r'Lifetime $\tau_{\rm sub}$ [s]', fontsize=13)
    ax.set_title('Silicate lifetime: CALIMA vs GD89', fontsize=13)
    ax.grid(alpha=0.25, which='both')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis='both', direction='in', labelsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path,
                format=os.path.splitext(filename)[1].lstrip('.') or 'png',
                dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print('Saved plot:', plot_path)
    print('Saved summary:', summary_path)
    print('Silicate closeness metrics:')
    print(
        f"  median|dex|={metrics['median_abs_dex']:.3f}, "
        f"mean|dex|={metrics['mean_abs_dex']:.3f}, "
        f"ratio=[{metrics['min_ratio']:.2e}, {metrics['max_ratio']:.2e}], "
        f"median ratio={metrics['median_ratio']:.2e}"
    )

    return {
        'plot_path': plot_path,
        'summary_path': summary_path,
        'metrics': metrics,
    }


_GD89_TDIST_DATASETS = {
    'graphite': {
        3.0: '3A_graphite_dPdlnT_GD89.csv',
        15.0: '15A_graphite_dPdlnT_GD89.csv',
        50.0: '50A_graphite_dPdlnT_GD89.csv',
    },
    'silicate': {
        3.5: '3.5A_silicate_dPdlnT_GD89.csv',
        15.0: '15A_silicate_dPdlnT_GD89.csv',
        50.0: '50A_silicate_dPdlnT_GD89.csv',
    },
}


def compute_gd89_temperature_distributions(
        material,
        G0=1.0,
        grain_sizes_angstrom=None,
        temperature_method='full',
        n_bins_full=300,
        n_bins_coarse=50,
        n_bins_fine=300,
        T_min=2.0,
        include_digitized=True,
        use_gd89_isrf=False):
    """Compute stochastic temperature distributions for GD89 reference sizes.

    Parameters
    ----------
    material : str
        ``'graphite'`` or ``'silicate'``.
    G0 : float
        Mathis ISRF scaling factor.
    grain_sizes_angstrom : sequence of float, optional
        Grain radii in Angstrom. If ``None``, uses available GD89
        digitized-size defaults for the chosen material.
    temperature_method : str
        ``'full'`` (default) or ``'adaptive'``.
    n_bins_full, n_bins_coarse, n_bins_fine, T_min
        Temperature-distribution controls forwarded to
        ``_temperature_distribution_result``.
    include_digitized : bool
        If ``True``, attach the matching GD89 digitized dP/dlnT data when
        available for each grain size.
    use_gd89_isrf : bool
        If ``True``, use exact digitized GD89 paper's Mathis et al. (1983) ISRF.

    Returns
    -------
    dict
        ``{'material': ..., 'G0': ..., 'distributions': {aA: {...}}}`` where
        each size entry includes ``T``, ``P``, ``dPdlnT`` and optional
        ``gd89_T``, ``gd89_dPdlnT``.
    """
    material = str(material).lower()
    if material not in ('graphite', 'silicate'):
        raise ValueError("material must be 'graphite' or 'silicate'.")

    if grain_sizes_angstrom is None:
        grain_sizes_angstrom = sorted(_GD89_TDIST_DATASETS.get(material, {}).keys())
    if len(grain_sizes_angstrom) == 0:
        raise ValueError(f'No GD89 grain sizes configured for material: {material}')

    wav_f, field = mathis_radiation_field_spectrum(G0=G0, use_gd89_isrf=use_gd89_isrf)
    wav_em = np.logspace(np.log10(0.1), np.log10(1000.0), 1000) * 1e-4

    out = {}
    dataset_map = _GD89_TDIST_DATASETS.get(material, {})

    for aA in grain_sizes_angstrom:
        aA = float(aA)
        a_cm = aA * 1e-8
        _, wav_xs, _, C_abs, _ = interpolate_cross_sections(material, a_cm * 1e4)

        res = _temperature_distribution_result(
            material, a_cm, wav_xs, C_abs, wav_f, field,
            wav_em=wav_em,
            method=temperature_method,
            n_bins_full=n_bins_full,
            n_bins_coarse=n_bins_coarse,
            n_bins_wide=n_bins_fine,
            n_bins_narrow=n_bins_fine,
            T_min=T_min,
        )

        T = np.asarray(res['T'], dtype=float)
        P = np.asarray(res['P'], dtype=float)
        dlnT = np.gradient(np.log(T))
        dPdlnT = np.zeros_like(P)
        good = dlnT > 0.0
        dPdlnT[good] = P[good] / dlnT[good]

        entry = {
            'a_angstrom': aA,
            'a_cm': a_cm,
            'T': T,
            'P': P,
            'dPdlnT': dPdlnT,
            'T_eq': float(res['T_eq']),
            'is_equilibrium': bool(res['is_equilibrium']),
            'delta_T': float(res['delta_T']),
            'T_range': tuple(res['T_range']),
        }

        if include_digitized and aA in dataset_map:
            data_path = os.path.join(_EXTERNAL_DATA_DIR, dataset_map[aA])
            if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
                print(f'Warning: GD89 digitized file is empty and will be skipped: {data_path}')
                out[aA] = entry
                continue
            try:
                gd = np.loadtxt(data_path, delimiter=',')
                gd = np.atleast_2d(gd)
                entry['gd89_T'] = np.asarray(gd[:, 0], dtype=float)
                entry['gd89_dPdlnT'] = np.asarray(gd[:, 1], dtype=float)
            except ValueError:
                print(f'Warning: Failed to parse GD89 digitized file and it will be skipped: {data_path}')

        out[aA] = entry

    return {
        'material': material,
        'G0': float(G0),
        'temperature_method': str(temperature_method),
        'distributions': out,
    }


def plot_gd89_temperature_distributions(
        material,
        G0=1.0,
        grain_sizes_angstrom=None,
        temperature_method='full',
        n_bins_full=300,
        n_bins_coarse=50,
        n_bins_fine=300,
        T_min=2.0,
        output_dir=None,
        filename=None,
        use_gd89_isrf=False,
        show=False):
    """Plot dP/dlnT for GD89 reference grain sizes for one material."""
    result = compute_gd89_temperature_distributions(
        material=material,
        G0=G0,
        grain_sizes_angstrom=grain_sizes_angstrom,
        temperature_method=temperature_method,
        n_bins_full=n_bins_full,
        n_bins_coarse=n_bins_coarse,
        n_bins_fine=n_bins_fine,
        T_min=T_min,
        include_digitized=True,
        use_gd89_isrf=use_gd89_isrf,
    )

    material = result['material']
    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'results')
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f'{material}_gd89_temperature_distributions.png'

    sizes = sorted(result['distributions'].keys())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sizes)))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0), dpi=180,
                           facecolor='w', edgecolor='k')

    for color, aA in zip(cmap, sizes):
        entry = result['distributions'][aA]
        ax.plot(entry['T'], entry['dPdlnT'], color=color, linewidth=2.0,
                label=f'CALIMA {material} {aA:g}A')

        if 'gd89_T' in entry:
            ax.scatter(entry['gd89_T'], entry['gd89_dPdlnT'],
                       color=color, marker='o', s=22,
                       edgecolors='none',
                       label=f'GD89 {material} {aA:g}A')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Dust temperature $T$ [K]', fontsize=13)
    ax.set_ylabel(r'$dP/d\ln T$', fontsize=13)
    ax.set_title(f'{material.capitalize()} temperature distributions vs GD89', fontsize=13)
    ax.grid(alpha=0.25, which='both')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis='both', direction='in', labelsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path,
                format=os.path.splitext(filename)[1].lstrip('.') or 'png',
                dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print('Saved plot:', plot_path)
    return {
        'plot_path': plot_path,
        'result': result,
    }


def plot_gd89_graphite_temperature_distributions(
        G0=1.0,
        grain_sizes_angstrom=(3.0, 15.0, 50.0),
        temperature_method='full',
        n_bins_full=300,
        n_bins_coarse=50,
        n_bins_fine=300,
        T_min=2.0,
        output_dir=None,
        filename='graphite_gd89_temperature_distributions.png',
        use_gd89_isrf=False,
        show=False):
    """Convenience wrapper for graphite GD89-size dP/dlnT plots."""
    return plot_gd89_temperature_distributions(
        material='graphite',
        G0=G0,
        grain_sizes_angstrom=grain_sizes_angstrom,
        temperature_method=temperature_method,
        n_bins_full=n_bins_full,
        n_bins_coarse=n_bins_coarse,
        n_bins_fine=n_bins_fine,
        T_min=T_min,
        output_dir=output_dir,
        filename=filename,
        use_gd89_isrf=use_gd89_isrf,
        show=show,
    )


def plot_gd89_silicate_temperature_distributions(
        G0=1.0,
        grain_sizes_angstrom=(3.5, 15.0, 50.0),
        temperature_method='full',
        n_bins_full=300,
        n_bins_coarse=50,
        n_bins_fine=300,
        T_min=2.0,
        output_dir=None,
        filename='silicate_gd89_temperature_distributions.png',
        use_gd89_isrf=False,
        show=False):
    """Convenience wrapper for silicate GD89-size dP/dlnT plots."""
    return plot_gd89_temperature_distributions(
        material='silicate',
        G0=G0,
        grain_sizes_angstrom=grain_sizes_angstrom,
        temperature_method=temperature_method,
        n_bins_full=n_bins_full,
        n_bins_coarse=n_bins_coarse,
        n_bins_fine=n_bins_fine,
        T_min=T_min,
        output_dir=output_dir,
        filename=filename,
        use_gd89_isrf=use_gd89_isrf,
        show=show,
    )


def compare_large_grain_timescales_wd00(
        grain_sizes_micron=(0.005, 0.01, 0.03, 0.1),
        silicate_temp_range=(1000.0, 2200.0),
        graphite_temp_range=(1500.0, 3500.0),
        n_points=150,
        output_dir=None,
        filename='large_grains_wd00_comparison.png',
        show=False):
    """Compare large-grain sublimation timescales to Waxman & Draine (2000).

    Computes CALIMA's full sublimation timescale t_sub = a / |da/dt| (which
    includes microcanonical/finite-system and surface tension/free-energy
    corrections) as a function of dust temperature T_dust and compares it
    to the bulk analytical approximation formulas used in Waxman & Draine
    (2000, WD00, Eqs. 3 and 4) for a range of grain sizes.

    Parameters
    ----------
    grain_sizes_micron : sequence of float
        Grain radii in micron to compare and plot.
    silicate_temp_range : tuple of (float, float)
        Temperature bounds [K] for Silicate grains.
    graphite_temp_range : tuple of (float, float)
        Temperature bounds [K] for Graphite grains.
    n_points : int
        Number of temperature grid points.
    output_dir : str, optional
        Target directory for saved plot. Defaults to repository 'results/'.
    filename : str
        Saved plot file name.
    show : bool
        If ``True``, call ``plt.show()`` after drawing.
    """
    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'results')
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), dpi=180,
                             facecolor='w', edgecolor='k')

    materials = ['silicate', 'graphite']
    temp_ranges = {
        'silicate': silicate_temp_range,
        'graphite': graphite_temp_range,
    }

    # Print nice header for terminal output
    print("=" * 78)
    print("SUBLIMATION TIMESCALE VS TEMPERATURE: CALIMA VS WAXMAN & DRAINE (2000)")
    print("=" * 78)

    for i, mat in enumerate(materials):
        ax = axes[i]
        t_bounds = temp_ranges[mat]
        T_grid = np.linspace(t_bounds[0], t_bounds[1], n_points)

        # Choose a color palette
        colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(grain_sizes_micron))) if mat == 'silicate' else plt.cm.viridis(np.linspace(0.1, 0.85, len(grain_sizes_micron)))

        print(f"\nMaterial: {mat.upper()}")
        print(f"{'Size [um]':10s} | {'Temp [K]':8s} | {'CALIMA [s]':11s} | {'WD00 [s]':11s} | {'Ratio'}")
        print("-" * 78)

        for a_mu, color in zip(grain_sizes_micron, colors):
            a_mu = float(a_mu)
            a_cm = a_mu * 1e-4
            a_5 = a_cm / 1e-5

            # Compute CALIMA timescales for each temperature
            t_calima = np.array([sublimation_timescale(mat, a_cm, t) for t in T_grid])

            # WD00 analytical formulas
            if mat == 'silicate':
                t_wd = 6.36e3 * a_5 * np.exp(68100.0 * (1.0 / T_grid - 1.0 / 1800.0))
                # For printing, pick 1500 K and 1800 K
                print_temps = [1500.0, 1800.0]
            else:  # graphite
                t_wd = 1.36 * a_5 * np.exp(81200.0 * (1.0 / T_grid - 1.0 / 3000.0))
                # For printing, pick 2400 K and 3000 K
                print_temps = [2400.0, 3000.0]

            # Plot curves
            ax.plot(T_grid, t_calima, color=color, linewidth=2.0,
                    label=r'{:g} $\mu$m (CALIMA)'.format(a_mu))
            ax.plot(T_grid, t_wd, color=color, linewidth=1.5, linestyle='--',
                    alpha=0.8, label=r'{:g} $\mu$m (WD00)'.format(a_mu))

            # Highlight print_temps stats
            for t_ref in print_temps:
                idx = np.argmin(np.abs(T_grid - t_ref))
                t_actual_k = T_grid[idx]
                tau_c = t_calima[idx]
                tau_w = t_wd[idx]
                ratio = tau_c / tau_w
                ratio_str = f"{ratio:10.4f}" if np.isfinite(ratio) else f"{'N/A':10s}"
                print(f"{a_mu:10.4f} | {t_actual_k:8.1f} | {tau_c:11.4e} | {tau_w:11.4e} | {ratio_str}")

        # Label subplots
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1e18)
        ax.set_xlabel(r'Dust temperature $T_{\rm dust}$ [K]', fontsize=12)
        ax.set_ylabel(r'Sublimation timescale $\tau_{\rm sub}$ [s]', fontsize=12)
        ax.set_title(f'{mat.capitalize()} lifetime vs temperature', fontsize=12)
        ax.grid(alpha=0.25, which='both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(which='both', axis='both', direction='in', labelsize=10)

        # Custom legend layout to keep it clean (T_K on left, types on right or simplified)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')

    fig.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\n" + "=" * 78)
    print(f"Saved comparison plot: {plot_path}")
    print("=" * 78)

    return plot_path


