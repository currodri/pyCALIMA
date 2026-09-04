import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pycalima.models.dust_charge.IM19_charging import grain_charge_dist,grain_mean_charge,grain_charge_probability
from pycalima.models.grain_size_config import get_optical_props_path
from pycalima.models.tools.radiation_fields import Draine_1978_isrf, Mathis83_radiation_field
from pycalima.models.dust_charge.shared_physics import (
    ionisation_potential_valence_vec as _ip_valence_vec,
    electron_affinity_graphite_vec as _ea_graphite_vec,
    electron_affinity_silicate_vec as _ea_silicate_vec,
    min_energy_ejection_vec as _emin_ejection_vec,
    photodetachment_energy_graphite_vec as _epdt_graphite_vec,
    photodetachment_energy_silicate_vec as _epdt_silicate_vec,
    min_photon_energy_vec as _hnu_min_vec,
    parameter_theta_vec as _theta_vec,
    escape_fraction_attempting_electrons_vec as _escape_frac_vec,
    photon_attenuation_length_graphite_vec as _la_graphite_vec,
    photon_attenuation_length_silicate_vec as _la_silicate_vec,
    Watson73_y1_vec as _watson_y1_vec,
    BT94_y0_graphite_vec as _y0_graphite_vec,
    y0_silicate_vec as _y0_silicate_vec,
    autoionisation_potential_graphite as _uait_graphite,
    autoionisation_potential_silicate as _uait_silicate,
    most_negative_allowed_charge_graphite as _zmin_graphite,
    most_negative_allowed_charge_silicate as _zmin_silicate,
    most_positive_allowed_charge as _zmax_allowed,
    electron_sticking_coefficient_graphite as _stick_graphite,
    electron_sticking_coefficient_silicate as _stick_silicate,
    DS87_J_function_scalar as _ds87_j_scalar,
    DS87_lambda_scalar as _ds87_lambda_scalar,
    DS87_J_function_vec as _ds87_j_vec,
    _coulomb_energy_over_a as _coulomb_e_over_a,
)
from pycalima.models.dust_radiation.dust_oppacity import read_dielectric_file, save_imn_file
from pycalima import _paths
from pycalima.models.grain_size_config import get_model_data_dir
from pycalima.plotting_style import use_calima_style


def _bpass_sed_dir():
    """Directory holding the BPASS v2.2.1 SED tables.

    These are ~GB of stellar-population spectra from a separate project
    (Dusty-PRISM) and are not redistributed with pyCALIMA. Point
    $CALIMA_SED_DIR at your own copy.
    """
    import os
    from pathlib import Path

    raw = os.environ.get("CALIMA_SED_DIR")
    if not raw:
        raise RuntimeError(
            "This routine needs the BPASS v2.2.1 SED tables, which are not "
            "bundled with pyCALIMA (they come from the Dusty-PRISM project and "
            "are far too large to ship). Set $CALIMA_SED_DIR to the directory "
            "containing them, e.g.\n"
            "    export CALIMA_SED_DIR=/path/to/bpass_v221_cha300"
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"$CALIMA_SED_DIR points at {path}, which is not a directory."
        )
    return str(path)

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _identity(func):
            return func
        return _identity

# CONSTANTS
graphite_work_function = 4.4 # [eV]
silicate_work_function = 8.0 # [eV]
me = 9.1093837015e-28         # Electron mass [g]
h_cgs = 6.62607015e-27        # Planck constant [erg s]
c_cgs = 2.99792458e10         # Speed of light [cm/s]
kb_cgs = 1.380649e-16         # Boltzmann constant [erg/K]
silicate_band_gap = 5.0 # [eV]
electron_escape_length = 1e-7 # [cm]
eV2erg = 1.602176634e-12  # Conversion factor from eV to erg

DS87_theta_nu = np.array([0.4203,0.5000,0.5823,0.6296,0.6621,0.6865,0.7560,0.8146])
DS87_nu = np.array([0.5,1,2,3,4,5,10,20])

PATH_OPTICS = str(get_optical_props_path())
_RADIATION_FIELD_LOGGED_ONCE = False
_EXTERNAL_DATA_DIR = str(_paths.get_external_data_path())


def dust_photoelectric_output_dir(*subdirs):
    """Directory for generated photoelectric-heating tables.

    Resolved on every call, not at import: it depends on $CALIMA_DATA and on
    the active configuration's model_name. The previous module-level
    ``os.makedirs`` ran at *import* time and raised PermissionError on a
    read-only install prefix.
    """
    return get_model_data_dir().joinpath('dust_photoelectric_heating_data', *subdirs)


def _photoelectric_output_path(path):
    if os.path.isabs(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    out_path = os.path.join(str(dust_photoelectric_output_dir()), path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def _external_data_path(path):
    return os.path.join(_EXTERNAL_DATA_DIR, path)


def _grain_output_label(a_cm=None, grain_label=None, a_micron=None):
    """Build a filesystem-safe grain label for filenames.

    If ``grain_label`` is provided (e.g. JSON bin id), use it directly.
    Otherwise build a cgs-based radius tag from ``a_cm``.
    """
    if grain_label is not None:
        label = str(grain_label).strip()
        if label:
            return label.replace('/', '_').replace(' ', '_')
    if a_cm is None and a_micron is not None:
        a_cm = float(a_micron) * 1e-4
    if a_cm is None:
        return 'grain'
    return f'a_{float(a_cm):.3e}_cm'


def _write_photoelectric_legacy_tables(out_dir, mode, size_tag, T_vals, gamma_vals, peh_log, rec_log):
    """Write the legacy Fortran-facing photoelectric tables with metadata headers.

    The grid file is omitted: axis values (log10(T) and log10(gamma)) are embedded
    directly into the rate files to match the charging-table convention.
    """
    heating_path = os.path.join(out_dir, f'dust_rates_peh_{size_tag}.dat')
    cooling_path = os.path.join(out_dir, f'dust_rates_rec_{size_tag}.dat')

    log_T = np.log10(T_vals)
    log_gamma = np.log10(gamma_vals)

    from pycalima.models.grain_size_config import get_header_lines
    header_lines = get_header_lines(
        title=f"Photoelectric heating/cooling rate table metadata (mode={mode})",
        script_name="models/dust_charge/dust_photoelectric_heating.py",
        bin_info=f"Dust Bin: {size_tag}",
        val_desc="Values: log10(rate [erg s^-1])",
        num_lines=6
    )

    # Write heating file with embedded axes (nT n_gamma, then temp line, gamma line, then data rows)
    with open(heating_path, 'w') as fh:
        fh.write('\n'.join(header_lines) + '\n')
        fh.write(f'{log_T.size} {log_gamma.size}\n')
        fh.write(' '.join(f'{value:.12e}' for value in log_T) + '\n')
        fh.write(' '.join(f'{value:.12e}' for value in log_gamma) + '\n')
        for row in np.asarray(peh_log, dtype=float).T:
            fh.write(' '.join(f'{value:.12e}' for value in row) + '\n')

    # Write cooling file similarly
    with open(cooling_path, 'w') as fh:
        fh.write('\n'.join(header_lines) + '\n')
        fh.write(f'{log_T.size} {log_gamma.size}\n')
        fh.write(' '.join(f'{value:.12e}' for value in log_T) + '\n')
        fh.write(' '.join(f'{value:.12e}' for value in log_gamma) + '\n')
        for row in np.asarray(rec_log, dtype=float).T:
            fh.write(' '.join(f'{value:.12e}' for value in row) + '\n')

    # No separate grid file returned anymore
    return None, heating_path, cooling_path


@njit(cache=True)
def _coulomb_energy_over_a_nb(Z, a):
    a_safe = a if a > 1e-300 else 1e-300
    return (4.8032047e-10 ** 2.0) * (Z + 1.0) / a_safe / 1.602176634e-12


@njit(cache=True)
def _ionisation_potential_valence_nb(W, Z, a):
    a_safe = a if a > 1e-300 else 1e-300
    return W + (4.8032047e-10 ** 2.0) / a_safe * ((Z + 0.5) + (Z + 2.0) * (0.3e-8 / a_safe)) / 1.602176634e-12


@njit(cache=True)
def _electron_affinity_graphite_nb(Z, a):
    a_safe = a if a > 1e-300 else 1e-300
    return 4.4 + (4.8032047e-10 ** 2.0) / a_safe * ((Z - 0.5) - (4e-8 / (a_safe + 7e-8))) / 1.602176634e-12


@njit(cache=True)
def _electron_affinity_silicate_nb(Z, a):
    a_safe = a if a > 1e-300 else 1e-300
    return 8.0 - 5.0 + (4.8032047e-10 ** 2.0) / a_safe * (Z - 0.5) / 1.602176634e-12


@njit(cache=True)
def _min_energy_ejection_nb(Z, a):
    if Z >= 0:
        return 0.0
    a_safe = a if a > 1e-300 else 1e-300
    att = 1.0 + (27e-8 / a_safe) ** 0.75
    return -(Z + 1.0) * (4.8032047e-10 ** 2.0) / (a_safe * att) / 1.602176634e-12


@njit(cache=True)
def _min_photon_energy_nb(IPV, Z, a):
    emin = _min_energy_ejection_nb(Z, a)
    if Z >= -1:
        return IPV
    return IPV + emin


@njit(cache=True)
def _parameter_theta_nb(E, Emin_ej, Z, a):
    if Z >= 0:
        return E - Emin_ej + _coulomb_energy_over_a_nb(Z, a)
    return E - Emin_ej


@njit(cache=True)
def _escape_fraction_nb(hnu, Emin_ej, Z, a):
    if Z < 0:
        return 1.0
    elow = -_coulomb_energy_over_a_nb(Z, a)
    ehigh = hnu - Emin_ej
    denom = (ehigh - elow) ** 3.0
    if abs(denom) < 1e-300:
        denom = 1e-300
    y2 = (ehigh ** 2.0) * (ehigh - 3.0 * elow) / denom
    if y2 < 0.0:
        return 0.0
    if y2 > 1.0:
        return 1.0
    return y2


@njit(cache=True)
def _attempting_integral_nb(hnu, Emin, Emin_ej, Z, a):
    if Z < 0:
        Elow = Emin
        Ehigh = Emin + hnu - Emin_ej
        Ei = Emin
        Ef = Ehigh
    else:
        Elow = -_coulomb_energy_over_a_nb(Z, a)
        Ehigh = hnu - Emin_ej
        Ei = 0.0
        Ef = Ehigh

    den = (Elow - Ehigh) ** 3
    if abs(den) < 1e-300:
        den = 1e-300
    A = Ef ** 2 * (6.0 * Ehigh * Elow - 4.0 * Elow * Ef - 4.0 * Ehigh * Ef + 3.0 * Ef ** 2) / (2.0 * den)
    B = Ei ** 2 * (6.0 * Ehigh * Elow - 4.0 * Elow * Ei - 4.0 * Ehigh * Ei + 3.0 * Ei ** 2) / (2.0 * den)
    return A - B


@njit(cache=True)
def _photon_attenuation_length_graphite_nb(wav, Imperp, Impar):
    l_inv = (4.0 * math.pi / wav) * ((2.0 / 3.0) * Imperp + (1.0 / 3.0) * Impar)
    if l_inv < 1e-300:
        l_inv = 1e-300
    return 1.0 / l_inv


@njit(cache=True)
def _photon_attenuation_length_silicate_nb(wav, Im):
    den = 4.0 * math.pi * Im
    if den < 1e-300:
        den = 1e-300
    return wav / den


@njit(cache=True)
def _watson_y1_nb(a, la, le):
    beta = a / la
    alpha = a / le + a / la
    num = (beta / alpha) ** 2.0 * (alpha ** 2 - 2.0 * alpha + 2.0 - 2.0 * math.exp(-alpha))
    den = beta ** 2 - 2.0 * beta + 2.0 - 2.0 * math.exp(-beta)
    if den < 1e-300:
        den = 1e-300
    return num / den


@njit(cache=True)
def _y0_graphite_nb(theta, W):
    w_safe = W if W > 1e-300 else 1e-300
    x = theta / w_safe
    if x < 0.0:
        x = 0.0
    x5 = x ** 5
    return (9e-3 * x5) / (1.0 + 3.7e-2 * x5)


@njit(cache=True)
def _y0_silicate_nb(theta, W):
    w_safe = W if W > 1e-300 else 1e-300
    x = theta / w_safe
    if x < 0.0:
        x = 0.0
    return 0.5 * x / (1.0 + 5.0 * x)


@njit(cache=True)
def _photodetachment_cross_nb(E, E_det, Z):
    x = (E - E_det) / 3.0
    if x < 0.0:
        return 0.0
    return 1.2e-17 * abs(Z) * x / (1.0 + (x * x) / 3.0) ** 2.0


@njit(cache=True)
def _photodetachment_energy_graphite_nb(Z, a):
    return _electron_affinity_graphite_nb(Z + 1, a) + _min_energy_ejection_nb(Z, a)


@njit(cache=True)
def _photodetachment_energy_silicate_nb(Z, a):
    return _electron_affinity_silicate_nb(Z + 1, a) + _min_energy_ejection_nb(Z, a)


@njit(cache=True)
def _compute_photoelectric_heating_graphite_numba(Z, a, energy_eV, wav_cm, I_E_surface, C_abs, Im_perp, Im_par):
    n = energy_eV.size
    Emin = _min_energy_ejection_nb(Z, a)
    IPV = _ionisation_potential_valence_nb(4.4, Z, a)
    Emin_ej = _min_photon_energy_nb(IPV, Z, a)

    Gamma = 0.0
    have_prev = False
    prev_x = 0.0
    prev_y = 0.0

    for i in range(n):
        E = energy_eV[i]
        if E < Emin_ej or E <= 0.0:
            y = 0.0
        else:
            theta = _parameter_theta_nb(E, Emin_ej, Z, a)
            y0 = _y0_graphite_nb(theta, 4.4)
            la = _photon_attenuation_length_graphite_nb(wav_cm[i], Im_perp[i], Im_par[i])
            y1 = _watson_y1_nb(a, la, 1e-7)
            y2 = _escape_fraction_nb(E, Emin_ej, Z, a)
            yld = y2 * min(y0 * y1, 1.0)
            if yld > 0.0 and y2 > 0.0:
                integral_fE = _attempting_integral_nb(E, Emin, Emin_ej, Z, a) / y2
                if integral_fE > 0.0:
                    y = yld * (I_E_surface[i] / E) * C_abs[i] * integral_fE
                else:
                    y = 0.0
            else:
                y = 0.0

        if have_prev:
            dx = E - prev_x
            Gamma += 0.5 * (prev_y + y) * dx
        prev_x = E
        prev_y = y
        have_prev = True

    if Z < 0:
        E_pdt = _photodetachment_energy_graphite_nb(Z, a)
        have_prev = False
        prev_x = 0.0
        prev_y = 0.0
        Gamma_det = 0.0
        for i in range(n):
            E = energy_eV[i]
            if E <= 0.0:
                y = 0.0
            else:
                sigma = _photodetachment_cross_nb(E, E_pdt, Z)
                y = sigma * (I_E_surface[i] / E) * (E - E_pdt + Emin)
            if have_prev:
                dx = E - prev_x
                Gamma_det += 0.5 * (prev_y + y) * dx
            prev_x = E
            prev_y = y
            have_prev = True
        Gamma += Gamma_det

    return Gamma


@njit(cache=True)
def _compute_photoelectric_heating_silicate_numba(Z, a, energy_eV, wav_cm, I_E_surface, C_abs, Im):
    n = energy_eV.size
    Emin = _min_energy_ejection_nb(Z, a)
    IPV = _ionisation_potential_valence_nb(8.0, Z, a)
    Emin_ej = _min_photon_energy_nb(IPV, Z, a)

    Gamma = 0.0
    have_prev = False
    prev_x = 0.0
    prev_y = 0.0

    for i in range(n):
        E = energy_eV[i]
        if E < Emin_ej or E <= 0.0:
            y = 0.0
        else:
            theta = _parameter_theta_nb(E, Emin_ej, Z, a)
            y0 = _y0_silicate_nb(theta, 8.0)
            la = _photon_attenuation_length_silicate_nb(wav_cm[i], Im[i])
            y1 = _watson_y1_nb(a, la, 1e-7)
            y2 = _escape_fraction_nb(E, Emin_ej, Z, a)
            yld = y2 * min(y0 * y1, 1.0)
            if yld > 0.0 and y2 > 0.0:
                integral_fE = _attempting_integral_nb(E, Emin, Emin_ej, Z, a) / y2
                if integral_fE > 0.0:
                    y = yld * (I_E_surface[i] / E) * C_abs[i] * integral_fE
                else:
                    y = 0.0
            else:
                y = 0.0

        if have_prev:
            dx = E - prev_x
            Gamma += 0.5 * (prev_y + y) * dx
        prev_x = E
        prev_y = y
        have_prev = True

    if Z < 0:
        E_pdt = _photodetachment_energy_silicate_nb(Z, a)
        have_prev = False
        prev_x = 0.0
        prev_y = 0.0
        Gamma_det = 0.0
        for i in range(n):
            E = energy_eV[i]
            if E <= 0.0:
                y = 0.0
            else:
                sigma = _photodetachment_cross_nb(E, E_pdt, Z)
                y = sigma * (I_E_surface[i] / E) * (E - E_pdt + Emin)
            if have_prev:
                dx = E - prev_x
                Gamma_det += 0.5 * (prev_y + y) * dx
            prev_x = E
            prev_y = y
            have_prev = True
        Gamma += Gamma_det

    return Gamma

# FUNCTIONS
def ionisation_potential_valence(W,Z,a):
    """Scalar cgs wrapper (a in cm, result in eV)."""
    return float(np.asarray(_ip_valence_vec(W, Z, a), dtype=float))

def plot_ionisation_potential(W):
    use_calima_style()
    Nc = np.linspace(10,80,20)
    a = (Nc/468.)**(1./3.) * 10 # [A]
    a = a / 10. # [nm]
    IPV0 = ionisation_potential_valence(W,0,a)
    IPV1 = ionisation_potential_valence(W,1,a)
    plt.figure(figsize=(8,6))
    plt.xlabel(r'$N_C$',fontsize=16)
    plt.ylabel(r'IPV [eV]',fontsize=16)
    plt.minorticks_on()
    plt.tick_params(labelsize=14, which='both',direction="in")
    plt.plot(Nc,IPV0,label='IPV(0)')
    plt.plot(Nc,IPV1,label='IPV(1)')
    data = np.loadtxt('IPV_graphite0_Draine.csv', delimiter=',')
    Nc_Draine = data[:, 0]
    IPV0_Draine = data[:, 1]
    plt.plot(Nc_Draine, IPV0_Draine, label='Draine IPV(0)', color='k', linestyle=':', linewidth=2)
    data = np.loadtxt('IPV_graphite1_Draine.csv', delimiter=',')
    Nc_Draine = data[:, 0]
    IPV1_Draine = data[:, 1]
    plt.plot(Nc_Draine, IPV1_Draine, label='Draine IPV(1)', color='k', linestyle=':', linewidth=2)

    plt.legend(fontsize=12,frameon=False)
    plt.savefig(_photoelectric_output_path('ionisation_potential.pdf'),format='pdf',dpi=300)

def plot_ionpot_vs_charge():
    use_calima_style()
    a = np.array([10,100]) # [nm]
    Z = np.arange(0,8000,100)

    plt.figure(figsize=(8,6))
    plt.xlabel(r'$Z$',fontsize=16)
    plt.ylabel(r'IPV [eV]',fontsize=16)
    plt.minorticks_on()
    plt.tick_params(labelsize=14, which='both',direction="in")
    for i in range(0,len(a)):
        IPV = ionisation_potential_valence(graphite_work_function,Z,a[i])
        plt.plot(Z,IPV,label=fr'$a={a[i]}$ nm')
    plt.legend(fontsize=12,frameon=False)
    plt.hlines(13.6,0,8000,colors='k',linestyles='dashed')
    plt.savefig(_photoelectric_output_path('ionisation_potential_vs_charge.pdf'),format='pdf',dpi=300)



def electron_affinity_graphite(W, Z, a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_ea_graphite_vec(Z, a), dtype=float))

def electron_affinity_silicate(W, Z, a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_ea_silicate_vec(Z, a), dtype=float))


def plot_electron_affinity():

    use_calima_style()
    Nc = np.linspace(5,60,100)
    a = (Nc/468.)**(1./3.) * 1e-7 # [cm]
    Z = 0
    EA = np.array([electron_affinity_graphite(graphite_work_function, Z, ai) for ai in a], dtype=float)
    plt.figure(figsize=(8,6))
    plt.ylim(-1,3)
    plt.xlabel(r'$N_C$',fontsize=16)
    plt.ylabel(r'$E_A$ [eV]',fontsize=16)
    plt.minorticks_on()
    plt.tick_params(labelsize=14, which='both',direction="in")
    plt.plot(Nc,EA,label='old')
    a = (Nc/468.)**(1./3.) * 10 # [A]
    Z = 0
    EA = graphite_work_function + 14.4 / a * ((Z-0.5) - 4 / (a+7))
    plt.plot(Nc,EA,label='Mine')
    plt.legend(fontsize=12,frameon=False)
    plt.savefig(_photoelectric_output_path('electron_affinity_graphite.pdf'),format='pdf',dpi=300)

def min_energy_ejection(Z,a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_emin_ejection_vec(Z, a), dtype=float))

def photodetachment_energy_graphite(Z,a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_epdt_graphite_vec(Z, a), dtype=float))

def photodetachment_energy_silicate(Z,a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_epdt_silicate_vec(Z, a), dtype=float))

def photodetachment_cross_section(E,E_det,Z):
    # Eq 20 Weingartner & Draine 2001
    x = (np.asarray(E, dtype=float) - E_det) / 3.0
    sigma = 1.2e-17 * abs(Z) * x / (1. + x**2./3.)**2.
    return np.where(x < 0.0, 0.0, sigma)

def min_photon_energy(IPV,Z,a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_hnu_min_vec(IPV, Z, a), dtype=float))

def parameter_theta(E,Emin_ej,Z,a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_theta_vec(E, Emin_ej, Z, a), dtype=float))

def attempting_electron_energy_dist(E,hnu,Emin,Emin_ej,Z,a):
    # Eq 10 Weingartner & Drain 2001
    if Z < 0:
        Elow = Emin
        Ehigh = Emin + hnu - Emin_ej
    else:
        Elow = - _coulomb_e_over_a(Z, a)
        Ehigh = hnu - Emin_ej

    return 6. * (E-Elow) * (Ehigh - E) / (Ehigh-Elow)**3.

def attempting_electron_energy_integral(hnu,Emin,Emin_ej,Z,a):
    if Z < 0:
        Elow = Emin
        Ehigh = Emin + hnu - Emin_ej
        Ei = Emin
        Ef = Ehigh
    else:
        Elow = - _coulomb_e_over_a(Z, a)
        Ehigh = hnu - Emin_ej
        Ei = 0
        Ef = Ehigh

    A = Ef**2 * (6.*Ehigh*Elow - 4.*Elow*Ef - 4.*Ehigh*Ef + 3*Ef**2) / (2.*(Elow-Ehigh)**3)
    B = Ei**2 * (6.*Ehigh*Elow - 4.*Elow*Ei - 4.*Ehigh*Ei + 3*Ei**2) / (2.*(Elow-Ehigh)**3)
    return A - B
def escape_fraction_attempting_electrons(hnu,Emin_ej,Z,a):
    # cgs scalar wrapper (a in cm)
    return float(np.asarray(_escape_frac_vec(hnu, Emin_ej, Z, a), dtype=float))

def photon_attenuation_length_graphite(wav,Imperp,Impar):
    # cgs scalar wrapper (wav in cm)
    return float(np.asarray(_la_graphite_vec(wav, Imperp, Impar), dtype=float))

def photon_attenuation_length_silicate(wav,Im):
    # cgs scalar wrapper (wav in cm)
    return float(np.asarray(_la_silicate_vec(wav, Im), dtype=float))

def Watson73_y1(a,la,le):
    # cgs scalar wrapper (all lengths in cm)
    return float(np.asarray(_watson_y1_vec(a, la, le), dtype=float))

def BT94_y0_graphite(theta,W):
    return float(np.asarray(_y0_graphite_vec(theta, W), dtype=float))

def y0_silicate(theta,W):
    return float(np.asarray(_y0_silicate_vec(theta, W), dtype=float))
def plot_dielectric_data(filename):
    """
    Plots the dielectric function data from the specified file.

    Parameters
    ----------
    filename : str
        Path to the input file.
    """
    use_calima_style()
    data = read_dielectric_file(filename)
    df = data['table']

    plt.figure(figsize=(10, 6))
    plt.plot(df['wavelength_um'], df['Re_n_minus_1'], label=r'Re($\epsilon$) - 1', color='blue')
    plt.plot(df['wavelength_um'], df['Im_n'], label=r'Im($\epsilon$)', color='orange')
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Dielectric Function')
    plt.title(rf"Dielectric Function for {data['icomp']} Dust (Radius: {data['radius_micron']} $\mu$m)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.savefig(_photoelectric_output_path(f'{filename}.png'),format='png',dpi=200)

def photoelectric_yield_graphite(W,Z,a,le,E,wav,Imperp,Impar):

    # 1. Compute IPV, Emin_ej
    IPV = ionisation_potential_valence(W,Z,a)
    Emin_ej = min_photon_energy(IPV,Z,a)
    if E < Emin_ej:
        return 0.0

    # 2. Compute theta and y0
    theta = parameter_theta(E,Emin_ej,Z,a)
    y0 = BT94_y0_graphite(theta,W)

    # 3. Compute the photon attenuation length
    la = photon_attenuation_length_graphite(wav,Imperp,Impar)

    # 4. Obtain y1
    y1 = Watson73_y1(a,la,le)

    # 5. Obtain y2
    y2 = escape_fraction_attempting_electrons(E,Emin_ej,Z,a)

    # 6. Compute the final yield
    Y = y2 * min(y0*y1,1.)
    if Y == 0.0:
        print(f"Yield is zero for E={E} eV, Z={Z}, a={a} micron.")
        print(f"    IPV: {IPV}, Emin_ej: {Emin_ej}, theta: {theta}, y0: {y0}, y1: {y1}, y2: {y2}, la: {la}")

    return Y

def photoelectric_yield_silicate(W,Z,a,le,E,wav,Im):

    # 1. Compute IPV, Emin_ej
    IPV = ionisation_potential_valence(W,Z,a)
    Emin_ej = min_photon_energy(IPV,Z,a)
    if E < Emin_ej:
        return 0.0

    # 2. Compute theta and y0
    theta = parameter_theta(E,Emin_ej,Z,a)
    y0 = y0_silicate(theta,W)

    # 3. Compute the photon attenuation length
    la = photon_attenuation_length_silicate(wav,Im)

    # 4. Obtain y1
    y1 = Watson73_y1(a,la,le)

    # 5. Obtain y2
    y2 = escape_fraction_attempting_electrons(E,Emin_ej,Z,a)

    # 6. Compute the final yield
    Y = y2 * min(y0*y1,1.)
    if Y > 1.0:
        print(f"Warning: Yield exceeds 1.0 for E={E} eV, Z={Z}, a={a} micron. Setting to 1.0.")

    return Y


def plot_photoelectric_yields(grain_types,a,Z,nE=100):

    # 1. Setup the figure
    use_calima_style()
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$Y$', fontsize=16)
    ax.set_xlabel(r'$E$ [eV]',fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_ylim([1e-2,1e0])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    
    # 2. Compute the grid of energies
    photon_energies = np.linspace(4,30,nE) # [eV]
    wavelengths = 1.2398e-4 / photon_energies # [cm]

    

    # 3. Loop over grain types
    for i in range(0, len(grain_types)):
        # 4. Interpolate the dielectric properties to the
        # requested wavelengths
        if grain_types[i] == 'graphite':
            data_perp = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10')
            data_par  = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10')
            wavelengths_um = wavelengths * 1e4
            Imperp = np.interp(wavelengths_um[::-1],data_perp['table'][::-1]['wavelength_um'], data_perp['table']['Im_n'][::-1])
            Impar = np.interp(wavelengths_um[::-1],data_par['table'][::-1]['wavelength_um'], data_par['table']['Im_n'][::-1])
            Imperp = Imperp[::-1]
            Impar = Impar[::-1]
        elif grain_types[i] == 'silicate':
            data = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/eps_suvSil')
            wavelengths_um = wavelengths * 1e4
            Im = np.interp(wavelengths_um[::-1],data['table'][::-1]['wavelength_um'], data['table']['Im_n'][::-1])
            Im = Im[::-1]

        # 5. Loop over energies and save the yields
        yields = np.zeros(nE)
        if grain_types[i] == 'graphite':
            for j in range(0, nE):
                yields[j] = photoelectric_yield_graphite(graphite_work_function,Z[i],
                                                         a[i],electron_escape_length,
                                                         photon_energies[j],wavelengths[j],
                                                         Imperp[j],Impar[j])
        elif grain_types[i] == 'silicate':
            for j in range(0, nE):
                yields[j] = photoelectric_yield_silicate(silicate_work_function,Z[i],
                                                         a[i],electron_escape_length,
                                                         photon_energies[j],wavelengths[j],
                                                         Im[j])
        
        # 6. Plot the yields
        ax.plot(photon_energies,yields,label=fr'{grain_types[i]}, $a={a[i]}$ $\mu$m, $Z={Z[i]}$')

    data = np.loadtxt(_external_data_path('Draine_yield_graphite.csv'), delimiter=',')
    energy_draine = data[:, 0]  # in eV
    yield_draine = data[:, 1]
    ax.plot(energy_draine, yield_draine, label='Draine 2011 - Graphite', color='k', linestyle=':', linewidth=2)

    data = np.loadtxt(_external_data_path('Draine_yield_silicate.csv'), delimiter=',')
    energy_draine = data[:, 0]  # in eV
    yield_draine = data[:, 1]
    ax.plot(energy_draine, yield_draine, label='Draine 2011 - Silicate', color='k', linestyle='--', linewidth=2)

    # 7. Legend and save plot
    ax.legend(loc='lower right',fontsize=12,frameon=False)
    fig.subplots_adjust(top=0.97,bottom=0.12,left=0.12,right=0.96,wspace=0,hspace=0)
    fig.savefig(_photoelectric_output_path('dust_photoelectric_yields.pdf'), format='pdf', dpi=300)


def compare_yield_functions(size_cm=4e-8, Z=0, E_min=4.0, E_max=30.0, nE=300, savefile=None):
    """
    Compare scalar yield functions in this module with the vectorized yields in
    `dust_charging` for graphite and silicate at the same grain size and Z.

    Parameters
    ----------
    size_cm : float
        Grain radius in cm.
    Z : int
        Grain charge to evaluate.
    E_min, E_max : float
        Photon energy range in eV.
    nE : int
        Number of energy samples.
    savefile : str or None
        If provided, save the figure to this path. Otherwise default filename is used.
    """
    use_calima_style()
    from pycalima.models.dust_charge.dust_charging import photoelectric_yield_graphite_vec, photoelectric_yield_silicate_vec

    # energy grid
    E = np.linspace(E_min, E_max, nE)  # eV
    # wavelengths in cgs
    wav_cm = 1.2398e-4 / E
    wav_micron = wav_cm * 1e4

    a_cm = float(size_cm)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # GRAPHITE
    data_perp = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10')
    data_par = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10')
    # interpolate Im to wav_micron
    Im_perp = np.interp(wav_micron, data_perp['table']['wavelength_um'][::-1], data_perp['table']['Im_n'][::-1])[::-1]
    Im_par = np.interp(wav_micron, data_par['table']['wavelength_um'][::-1], data_par['table']['Im_n'][::-1])[::-1]

    # scalar yields (graphite)
    Y_scalar_g = np.zeros_like(E)
    for i, Ei in enumerate(E):
        Y_scalar_g[i] = photoelectric_yield_graphite(graphite_work_function, Z, a_cm, electron_escape_length,
                               Ei, wav_cm[i], Im_perp[i], Im_par[i])

    # vectorized yields (graphite) in cgs
    Y_vec_g = photoelectric_yield_graphite_vec(graphite_work_function, np.array([Z]), a_cm, electron_escape_length, E, wav_cm, Im_perp, Im_par)
    # Y_vec_g shape [N_E, N_Z]
    Y_vec_g = Y_vec_g[:, 0]

    ax = axs[0]
    ax.plot(E, Y_scalar_g, label='scalar (old)', color='C0')
    ax.plot(E, Y_vec_g, label='vectorized', color='C1', linestyle='--')
    ax.set_xlabel('Photon energy (eV)')
    ax.set_ylabel('Yield Y')
    ax.set_title(f'Graphite a={size_cm:.3e} cm, Z={Z}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, ls=':')

    # SILICATE
    data_sil = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/eps_suvSil')
    Im_sil = np.interp(wav_micron, data_sil['table']['wavelength_um'][::-1], data_sil['table']['Im_n'][::-1])[::-1]

    # scalar yields (silicate)
    Y_scalar_s = np.zeros_like(E)
    for i, Ei in enumerate(E):
        Y_scalar_s[i] = photoelectric_yield_silicate(silicate_work_function, Z, a_cm, electron_escape_length,
                               Ei, wav_cm[i], Im_sil[i])

    # vectorized yields (silicate)
    Y_vec_s = photoelectric_yield_silicate_vec(silicate_work_function, np.array([Z]), a_cm, electron_escape_length, E, wav_cm, Im_sil)
    Y_vec_s = Y_vec_s[:, 0]

    ax = axs[1]
    ax.plot(E, Y_scalar_s, label='scalar (old)', color='C0')
    ax.plot(E, Y_vec_s, label='vectorized', color='C1', linestyle='--')
    ax.set_xlabel('Photon energy (eV)')
    ax.set_ylabel('Yield Y')
    ax.set_title(f'Silicate a={size_cm:.3e} cm, Z={Z}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, ls=':')

    fig.tight_layout()
    out = _photoelectric_output_path(savefile or f'yield_comparison_{size_cm:.3e}cm_Z{Z}.pdf')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('Saved yield comparison to', out)
    return out

def compute_photoelectric_heating_rate(args):

    Z,a,radiation_field,grain_type,Im,C_abs = args

    # Fast path: execute the expensive per-energy loop in Numba when inputs are numeric arrays.
    if _NUMBA_AVAILABLE:
        try:
            rf = np.asarray(radiation_field, dtype=float)
            cab = np.asarray(C_abs, dtype=float)
            if rf.ndim == 2 and rf.shape[1] >= 3 and cab.ndim == 1 and cab.size == rf.shape[0]:
                e_arr = np.ascontiguousarray(rf[:, 0], dtype=np.float64)
                wav_arr = np.ascontiguousarray(rf[:, 1], dtype=np.float64)
                i_arr = np.ascontiguousarray(rf[:, 2], dtype=np.float64)
                c_arr = np.ascontiguousarray(cab, dtype=np.float64)

                gtype = str(grain_type).lower()
                if gtype == 'graphite':
                    im = np.asarray(Im, dtype=float)
                    if im.ndim == 2 and im.shape[1] >= 2 and im.shape[0] == rf.shape[0]:
                        im_perp = np.ascontiguousarray(im[:, 0], dtype=np.float64)
                        im_par = np.ascontiguousarray(im[:, 1], dtype=np.float64)
                        return float(_compute_photoelectric_heating_graphite_numba(int(Z), float(a), e_arr, wav_arr, i_arr, c_arr, im_perp, im_par))
                elif gtype == 'silicate':
                    im = np.asarray(Im, dtype=float)
                    if im.ndim == 1 and im.size == rf.shape[0]:
                        im1 = np.ascontiguousarray(im, dtype=np.float64)
                        return float(_compute_photoelectric_heating_silicate_numba(int(Z), float(a), e_arr, wav_arr, i_arr, c_arr, im1))
        except Exception:
            # Fall back to the original Python implementation if fast-path checks fail.
            pass

    # 1. Compute the minimum energy for ejection
    Emin = min_energy_ejection(Z,a)
    if grain_type == 'graphite':
        IPV = ionisation_potential_valence(graphite_work_function,Z,a)
    elif grain_type == 'silicate':
        IPV = ionisation_potential_valence(silicate_work_function,Z,a)
    Emin_ej = min_photon_energy(IPV,Z,a)

    # 2. Loop over the photon energies
    dGamma = np.zeros(radiation_field.shape[0])
    yields = np.zeros(radiation_field.shape[0])
    intfE  = np.zeros(radiation_field.shape[0])
    for i in range(radiation_field.shape[0]):
        # if (radiation_field[i,0] < 13.6 or radiation_field[i,0] > 6):
        # 2. Compute the photoelectric yield for the photon energy
        if grain_type == 'graphite':
            yield_i = photoelectric_yield_graphite(graphite_work_function,Z,a,
                                                electron_escape_length,radiation_field[i,0],
                                                radiation_field[i,1],Im[i,0],Im[i,1])
        elif grain_type == 'silicate':
            yield_i = photoelectric_yield_silicate(silicate_work_function,Z,a,
                                                electron_escape_length,radiation_field[i,0],
                                                radiation_field[i,1],Im[i])
        yields[i] = yield_i
        # if yield_i == 0.: print("Yield:",yield_i,"Z:",Z,"a:",a,"E:",radiation_field[i,0])
        if yield_i > 0.0:
            # 3. Compute the integral of the photo-electron energy distribution
            y2 = escape_fraction_attempting_electrons(radiation_field[i,0],Emin_ej,Z,a)
            integral_fE = attempting_electron_energy_integral(radiation_field[i,0],Emin,Emin_ej,Z,a)/y2
            if integral_fE > 0.0:
                intfE[i] = integral_fE
                # 4. Finally compute the injected power
                dGamma[i] = yield_i * (radiation_field[i,2] / radiation_field[i,0]) * C_abs[i] * integral_fE
    # if all(dGamma == 0.0):
    #     print("Warning: No photoelectric heating for Z=",Z,"a=",a)
    #     print("Yields:",yields)
    #     print("Integral fE:",intfE)
    #     print("Radiation field:",radiation_field[:,0])
    #     print("C_abs:",C_abs)
    #     print("IPV:",IPV,"Emin_ej:",Emin_ej,"Emin:",Emin)
    # 5. Integrate over the photon energies to obtain the total heating rate
    Gamma = np.trapezoid(dGamma, radiation_field[:,0])

    # 6. Compute the contribution from photodetachment
    if Z < 0:
        if grain_type == 'graphite':
            E_pdt = photodetachment_energy_graphite(Z,a)
        elif grain_type == 'silicate':
            E_pdt = photodetachment_energy_silicate(Z,a)
        dGamma_det = photodetachment_cross_section(radiation_field[:,0],E_pdt,Z) * \
            (radiation_field[:,2] / radiation_field[:,0]) * (radiation_field[:,0] - E_pdt + Emin)

        Gamma_det = np.trapezoid(dGamma_det, radiation_field[:,0])
        Gamma += Gamma_det

    return Gamma 


def compute_photoelectric_heating_rate_single_bin(Z, a_cm, E_eV, I_E_surface,
                                                 grain_type='graphite',
                                                 C_abs_cm2=None, Im_val=None,
                                                 Imperp_val=None, Impar_val=None,
                                                 yield_params=None):
    """
    Compute the photoelectric heating contribution for a single grain charge
    state and a single radiation bin (energy E_eV and intensity I_E_surface).

    This helper will interpolate optical cross sections and dielectric
    imaginary parts for the requested grain size if scalar values are not
    provided. It returns the heating power (erg / s) contributed by that
    radiation bin for the grain.

    Parameters
    ----------
    Z : int
        Grain charge state (unitless).
    a_cm : float
        Grain radius in cm.
    E_eV : float
        Photon energy in eV for the bin.
    I_E_surface : float
        Spectral intensity at E (erg / s / cm^2 / eV) (surface-integrated / per-area).
    grain_type : str
        'graphite' or 'silicate'.
    a_cm : float
        Grain radius in cm. Cross sections are interpolated using the equivalent micron size.
        If provided and C_abs_cm2 is None, the function
        will interpolate the absorption cross section for this grain size.
    C_abs_cm2 : float or None
        Absorption cross section at the bin wavelength in cm^2. If None, the
        function will compute it from the optical tables using `a_micron`.
    Im_val, Imperp_val, Impar_val : float or None
        Imaginary parts of the dielectric at the bin wavelength. For
        silicate supply `Im_val`; for graphite supply `Imperp_val` and
        `Impar_val`. If missing, the function will read dielectric files and
        interpolate to the requested wavelength (requires `a_micron` or
        access to the dielectric tables in the repo).

    Returns
    -------
    Gamma_bin : float
        Photoelectric heating power contributed by this bin (erg / s).
    info : dict
        Diagnostic components: {'yield': Y, 'integral_fE': integral_fE, 'C_abs_cm2': C_abs_cm2}
    """
    # ensure numeric scalars
    Z = int(Z)
    a_cm = float(a_cm)
    a_micron = a_cm * 1e4
    E = float(E_eV)
    I_E = float(I_E_surface)

    # If C_abs not provided, attempt to interpolate from cross-section tables
    if C_abs_cm2 is None:
        from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections
        _, wav_cs, _, C_abs_cs, _ = interpolate_cross_sections(grain_type, a_micron)
        # wav_cs is an array of wavelengths (cm) per interpolate_cross_sections contract
        # convert to microns for the optical_E relation used elsewhere
        # optical_E (eV) = 1.2398 / (wav_um)
        optical_E = 1.2398 / (wav_cs * 1e4)
        C_abs_cm2 = float(np.interp(E, optical_E, C_abs_cs))

    # Obtain dielectric imaginary parts if not provided
    wav_cm = 1.2398e-4 / E
    wav_micron = wav_cm * 1e4
    if grain_type.lower().startswith('gra'):
        if Imperp_val is None or Impar_val is None:
            data_perp = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10')
            data_par = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10')
            # tables often stored with descending wavelengths; reverse to ensure increasing
            wav_tab_perp = np.asarray(data_perp['table']['wavelength_um'][::-1], dtype=float)
            Im_tab_perp = np.asarray(data_perp['table']['Im_n'][::-1], dtype=float)
            wav_tab_par = np.asarray(data_par['table']['wavelength_um'][::-1], dtype=float)
            Im_tab_par = np.asarray(data_par['table']['Im_n'][::-1], dtype=float)
            Imperp_val = float(np.interp(wav_micron, wav_tab_perp, Im_tab_perp))
            Impar_val = float(np.interp(wav_micron, wav_tab_par, Im_tab_par))
        # call scalar yield function
        Y = photoelectric_yield_graphite(graphite_work_function, Z, a_cm, electron_escape_length, E, wav_cm, Imperp_val, Impar_val)
    else:
        # silicate
        if Im_val is None:
            data = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/eps_suvSil')
            wav_tab = np.asarray(data['table']['wavelength_um'][::-1], dtype=float)
            Im_tab = np.asarray(data['table']['Im_n'][::-1], dtype=float)
            Im_val = float(np.interp(wav_micron, wav_tab, Im_tab))
        Y = photoelectric_yield_silicate(silicate_work_function, Z, a_cm, electron_escape_length, E, wav_cm, Im_val)

    # If yield is zero, no heating
    if Y <= 0.0:
        return 0.0, {'yield': float(Y), 'integral_fE': 0.0, 'C_abs_cm2': float(C_abs_cm2)}

    # compute energy-ejection parameters
    Emin = min_energy_ejection(Z, a_cm)
    if grain_type.lower().startswith('gra'):
        IPV = ionisation_potential_valence(graphite_work_function, Z, a_cm)
    else:
        IPV = ionisation_potential_valence(silicate_work_function, Z, a_cm)
    Emin_ej = min_photon_energy(IPV, Z, a_cm)

    # escape fraction and integral of attempting-electron distribution
    y2 = escape_fraction_attempting_electrons(E, Emin_ej, Z, a_cm)
    if y2 == 0.0:
        integral_fE = 0.0
    else:
        integral_fE = attempting_electron_energy_integral(E, Emin, Emin_ej, Z, a_cm) / y2
        if integral_fE < 0.0:
            integral_fE = 0.0

    # main heating contribution for the bin
    # I_E_surface is erg / s / cm^2 / eV, E in eV, C_abs_cm2 in cm^2
    print(I_E/E)
    Gamma_bin = Y * (I_E / E) * float(C_abs_cm2) * float(integral_fE)

    # photodetachment contribution for negative grains
    if Z < 0:
        if grain_type.lower().startswith('gra'):
            E_pdt = photodetachment_energy_graphite(Z, a_cm)
        else:
            E_pdt = photodetachment_energy_silicate(Z, a_cm)
        sigma_pd = photodetachment_cross_section(E, E_pdt, Z)
        if sigma_pd > 0.0:
            # convert sigma from cm^2? photodetachment_cross_section returns cm^2 in this module
            dGamma_det = sigma_pd * (I_E / E) * (E - E_pdt + Emin)
            Gamma_bin += float(dGamma_det)

    # compute the photon attenuation length
    if grain_type.lower().startswith('gra'):
        la = photon_attenuation_length_graphite(wav_cm, Imperp_val, Impar_val)
    else:
        la = photon_attenuation_length_silicate(wav_cm, Im_val)

    info = {'IPV': float(IPV),'Emin': float(Emin),'Emin_ej': float(Emin_ej), 'la': float(la), 'yield': float(Y), 'integral_fE': float(integral_fE), 'C_abs_cm2': float(C_abs_cm2)}
    return float(Gamma_bin), info


def compute_recombination_cooling_rate(args):
    Z,a,ne,T,grain_type = args

    ltilde = _ds87_lambda_scalar(Z,-1.,a,T)
    if grain_type == 'graphite':
        s = _stick_graphite(Z,a)
    elif grain_type == 'silicate':
        s = _stick_silicate(Z,a)
    recomb_rate = np.pi * a**2 * ne * s * np.sqrt(8. * kb_cgs * T / (np.pi * me)) * ltilde * kb_cgs * T
    return recomb_rate


def compute_autoionisation_cooling_rate(args):
    Zmin,prob_Zmin,a,ne,T,grain_type = args
    if grain_type == 'graphite':
        EA = electron_affinity_graphite(graphite_work_function,Zmin,a)
    elif grain_type == 'silicate':
        EA = electron_affinity_silicate(silicate_work_function,Zmin,a)
    Jtilde = _ds87_j_scalar(Zmin,-1.,a,T)
    autoion_rate = np.pi * a**2 * ne * prob_Zmin * np.sqrt(8. * kb_cgs * T / (np.pi * me)) *\
        Jtilde * EA * eV2erg
    return autoion_rate

def DS87_lambda_function(Z,q,a,T, return_tau=False):
    """
        Numeric implementation of the DS87 lambda-tilde function in cgs.
        Units contract: a in cm, T in K, Z/q dimensionless.
    """
    # numeric constants (cgs)
    e_statC = 4.8032047e-10   # statcoulomb

    Z = np.asarray(Z, dtype=float)
    q = np.asarray(q, dtype=float)

    # compute nu = Z / q (broadcasting-aware)
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = Z / q

    a_cm = np.asarray(a, dtype=float)

    # denominator: (q * e_statC)^2 -> handle broadcasting and avoid zero-division
    denom = (q ** 2) * (e_statC ** 2)
    tau = (a_cm * kb_cgs * T) / np.maximum(denom, 1e-300)

    try:
        tau = np.broadcast_to(tau, np.shape(nu))
    except Exception:
        tau = np.asarray(tau)

    tau_safe = np.maximum(tau, 1e-300)

    # allocate output and compute branch-wise
    ltilde = np.zeros_like(nu, dtype=float)

    nu_zero = (nu == 0.0)
    nu_neg = (nu < 0.0)
    nu_pos = (nu > 0.0)

    if np.any(nu_zero):
        ltilde[nu_zero] = 2.0 + 1.5 * np.sqrt(np.pi / (2.0 * tau_safe[nu_zero]))

    if np.any(nu_neg):
        tn = tau_safe[nu_neg]
        nun = nu[nu_neg]
        inner = np.maximum(tn - nun, 1e-300)
        ltilde[nu_neg] = (2.0 - nun / tn) * (1.0 + 1.0 / np.sqrt(inner))

    if np.any(nu_pos):
        tp = tau_safe[nu_pos]
        nup = np.maximum(nu[nu_pos], 1e-300)
        theta_nu = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
        ltilde[nu_pos] = (2.0 + nup / tp) * (1.0 + 1.0 / np.sqrt(1.5 / tp + 3.0 * nup)) * np.exp(-theta_nu / tp)

    # if inputs were scalar, return scalar
    if np.isscalar(Z) and np.isscalar(q):
        return float(ltilde)
    if return_tau:
        return ltilde, tau
    else:
        return ltilde

def DS87_J_function(Z,q,a,T):
    """
        Scalar-friendly cgs wrapper around dust_charging DS87_J_function_vec.
        a is grain radius in cm.
    """
    out = _ds87_j_vec(Z, q, a, T)
    if np.isscalar(Z) and np.isscalar(q):
        return float(np.asarray(out, dtype=float))
    return np.asarray(out, dtype=float)

def plot_DS87_lambda(Z,q,a,Tmin,Tmax,nT=100):
    """
    Plot the DS87 lambda function for a range of temperatures.
    
    Parameters
    ----------
    Z : float
        Charge of the grain.
    q : float
        Charge of the electron.
    a : float
        Grain size in cm.
    Tmin : float
        Minimum temperature in K.
    Tmax : float
        Maximum temperature in K.
    nT : int, optional
        Number of temperature points to compute (default is 100).
    """
    use_calima_style()
    import matplotlib.pyplot as plt

    temperatures = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    data = [DS87_lambda_function(Z,q,a,T,return_tau=True) for T in temperatures]
    ltilde_values = np.array([d[0] for d in data])
    tau = np.array([d[1] for d in data])

    data = np.loadtxt(_external_data_path('lambda_0.csv'), delimiter=',')
    Nc_Draine = data[:, 0]
    IPV0_Draine = data[:, 1]
    

    plt.figure(figsize=(8, 6))
    plt.plot(tau, ltilde_values, label=f'Z={Z}, q={q}, a={a} micron')
    plt.plot(Nc_Draine, IPV0_Draine, label='Draine IPV(0)', color='k', linestyle=':', linewidth=2)
    plt.xlabel(r'$\tau$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-2,1e3)
    plt.xlim(1e-2,1e3)
    plt.ylabel(r'$\tilde{\lambda}$')
    plt.title('DS87 Lambda Function')
    plt.legend()
    plt.grid()
    plt.savefig(_photoelectric_output_path('DS87_lambda_function.pdf'), format='pdf', dpi=300)

def autoionisation_potential_graphite(a):
    return float(np.asarray(_uait_graphite(a), dtype=float))

def autoionisation_potential_silicate(a):
    return float(np.asarray(_uait_silicate(a), dtype=float))

def most_negative_allowed_charge_graphite(a):
    return float(np.asarray(_zmin_graphite(a), dtype=float))

def most_negative_allowed_charge_silicate(a):
    return float(np.asarray(_zmin_silicate(a), dtype=float))

def electron_sticking_coefficient_graphite(Z,a):
    return float(np.asarray(_stick_graphite(Z, a), dtype=float))

def electron_sticking_coefficient_silicate(Z,a):
    return float(np.asarray(_stick_silicate(Z, a), dtype=float))

def electron_sticking_coefficient_Mengel2025(Z,a,T):
    from unyt import kb,m,K,statC
    s_e0 = 0.26
    s_e = s_e0 #* np.exp(-abs(Z)*(4.8032047e-10*statC)**2./(a*1e-9*m*kb*T*K))
    print('s',(4.8032047e-10*statC)**2./(a*1e-9*m*kb*T*K))
    return s_e

def most_positive_allowed_charge(a,W,Emax):
    return float(np.asarray(_zmax_allowed(a, W, Emax), dtype=float))

def plot_peh_vs_recombination(grain_types,a,G0,ne,Tmin,Tmax,nT=100,radiation_model='Draine', use_equilibrium=False):
    """
    Plot the photoelectric heating rate vs. recombination cooling rate for different grain types.
    
    Parameters
    ----------
    grain_types : list of str
        List of grain types (e.g., ['graphite', 'silicate']).
    a : list of float
        List of grain sizes in microns.
    G0 : float
        Radiation field strength in units of Draine field.
    ne : float
        Electron density in cm^-3.
    Tmin : float
        Minimum temperature in K.
    Tmax : float
        Maximum temperature in K.
    nT : int, optional
        Number of temperature points to compute (default is 100).
    radiation_model : str, optional
        Radiation model to use ('Draine' or 'Mathis') (default is 'Draine').
    """
    use_calima_style()
    from astropy.table import Table
    from pycalima.models.PAH_charge.PAH_photoelectric_heating import blackbody_radiation
    from pycalima.models.dust_radiation.dust_emission import compute_cross_sections
    from unyt import nm,m,cm,eV,J,s,h,c,erg,K,kb
    import concurrent.futures
    from tqdm import tqdm

    
    # 1. Setup the figure
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$\Gamma_{\rm PEH},\Lambda_{\rm rec}$ [erg s$^{-1}$]', fontsize=16)
    ax.set_xlabel(r'$T$ [K]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    
    # 2. Compute the grid of temperatures
    temperatures = np.linspace(Tmin,Tmax,nT)

    # 3. Get the radiation field model
    if radiation_model == 'Draine':
        draine_data = np.loadtxt(_external_data_path('draine1978.dat'))
        wavelength_nm = np.asarray(draine_data[:, 0], dtype=float)
        photon_flux = np.asarray(Draine_1978_isrf(wavelength_nm), dtype=float)
        rad_field = np.column_stack([wavelength_nm, photon_flux * wavelength_nm * eV2erg])
        rad_name = 'Draine 1978'
        rad_color = '#BBD8B3'
        linestyle= ':'
    elif radiation_model == 'Habing':
        draine_data = np.loadtxt(_external_data_path('draine1978.dat'))
        wavelength_nm = np.asarray(draine_data[:, 0], dtype=float)
        photon_flux = np.asarray(Draine_1978_isrf(wavelength_nm), dtype=float) / 1.7
        rad_field = np.column_stack([wavelength_nm, photon_flux * wavelength_nm * eV2erg])
        rad_name = 'Habing 1968'
        rad_color = '#F3B61F'
        linestyle= ':'
    elif radiation_model == 'Mathis':
        # Use the analytic Mathis (1983) functional form from Mathis83_radiation_field
        # Mathis83_radiation_field(E) returns u_E in units erg / cm^3 / eV.
        # To be consistent with other radiation models (which return
        # wavelength [nm] and I_lambda [erg / cm^2 / s / nm / sr]) we convert
        # the functional u_E -> I_lambda per nm per sr and return a 2-column array
        # [wavelength_nm, I_lambda].
        E = np.linspace(0.1, 13.6, 1000)
        u_E = np.asarray([Mathis83_radiation_field(e) for e in E])
        # constants
        h_SI = 6.62607015e-34
        eV2J = 1.602176634e-19
        c_SI = 2.99792458e8
        c_cgs = 2.99792458e10

        # dnu/dE (Hz per eV)
        dnu_dE = eV2J / h_SI

        # u_nu (erg cm^-3 Hz^-1) = u_E / (dnu/dE)
        u_nu = u_E / dnu_dE

        # I_nu (erg cm^-2 s^-1 Hz^-1 sr^-1) = c / (4π) * u_nu  [using c in cm/s]
        I_nu = (c_cgs / (4.0 * np.pi)) * u_nu

        # convert to I_lambda: I_lambda = I_nu * c / lambda^2
        E_J = E * eV2J
        lambda_m = h_SI * c_SI / E_J
        lambda_cm = lambda_m * 100.0
        I_lambda_per_cm = I_nu * c_cgs / (lambda_cm ** 2)

        # convert per cm to per nm: 1 nm = 1e-7 cm -> I_per_nm = I_per_cm * 1e-7
        I_lambda_per_nm = I_lambda_per_cm * 1e-7

        wavelength_nm = 1239.84193 / E
        rad_field = np.column_stack([wavelength_nm, I_lambda_per_nm])
    elif radiation_model == 'Mathis_file':
        # Explicit file-based Mathis (legacy behaviour). Returns the raw file columns
        mathis1983 = Table.read('../photoelectric-heating/ISRF/mathis1983.txt', format='ascii')
        rad_field = np.column_stack([mathis1983['col1'], mathis1983['col2']])
        rad_name = 'Mathis+1983'
        rad_color = '#A29F15'
        linestyle= ':'
    elif radiation_model == 'HD200775':
        HD200775 = Table.read('../photoelectric-heating/stars/HD200775_RF.txt', format='ascii')
        rad_field = np.column_stack([HD200775['col1'],HD200775['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'HD200775'
        rad_color = '#510D0A'
        linestyle= '--'
    elif radiation_model[:2] == 'BB':
        T_star = float(radiation_model[2:])
        # Obtain the black body radiation field in units of erg cm-2 s-1 nm-1 sr-1 for the 
        # given temperature
        BB = blackbody_radiation(T_star, 23.0, 500, num_points=1000)
        rad_field = np.column_stack([BB[0].to('nm').d,BB[1].to('erg/cm**2/s/nm').d])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'BB $T_{\star}=$'+str(int(T_star))+' K'
        rad_color = '#256EFF'
        linestyle= '--'
    elif radiation_model == 'O6V':
        O6V = Table.read('../photoelectric-heating/stars/kp00_40000', format='ascii')
        rad_field = np.column_stack([O6V['col1'],O6V['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'O6V'
        rad_color = '#C33149'
        linestyle= '-.'
    elif radiation_model == 'B0V':
        B0V = Table.read('../photoelectric-heating/stars/kp00_30000', format='ascii')
        rad_field = np.column_stack([B0V['col1'],B0V['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'B0V'
        rad_color = '#4B543B'
        linestyle= '-.'
    elif radiation_model == 'A0':
        A0 = Table.read('../photoelectric-heating/stars/kp00_10000', format='ascii')
        rad_field = np.column_stack([A0['col1'],A0['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'A0'
        rad_color = '#533A71'
        linestyle= '-.'
    elif radiation_model == 'BPASS_veryyoung_lowz':
        from pycalima.models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 0.01 # 10 Myr
        fixed_metallicity = 0.0002 # 0.01 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=10$ Myr, $Z=0.01Z_{\odot}$)'
        rad_color = '#258EA6'
        linestyle= '-'
    elif radiation_model == 'BPASS_young_midz':
        from pycalima.models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 0.1 # 0.1 Gyr
        fixed_metallicity = 0.01 # 0.5 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=0.5Z_{\odot}$)'
        rad_color = '#F75590'
        linestyle= '-'
    elif radiation_model == 'BPASS_old_highz':
        from pycalima.models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 1 # 1 Gyr
        fixed_metallicity = 0.02 # 1 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=Z_{\odot}$)'
        rad_color = "#79513E"
        linestyle= '-'

    wavelength_intensity = rad_field[:,1] # in erg cm-2 s-1 nm-1 sr-1
    wavelength = rad_field[:,0] # in nm
    I_rad = wavelength_intensity / (h * c / (wavelength * nm)).to('erg').d
    E = 1.2398 / (wavelength[::-1]*1e-3)
    I = I_rad[::-1] * cm**-2/s/nm
    F = I * E * eV
    f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
    I = f.to('erg/s/cm**2/eV').d 
    G0 = np.trapezoid(2.*np.pi*f.to('W/m**2/eV').d[(E<=13.6)&(E>=5.17)],E[(E<=13.6)&(E>=5.17)]) / 1.68e-6
    rad_field = np.column_stack([E,wavelength[::-1],I])

    # 3. Loop over grain types
    for i in range(0, len(grain_types)):
        # 4. Interpolate the dielectric properties to the requested wavelengths
        if grain_types[i] == 'graphite':
            data_perp = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10')
            data_par  = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10')
            Imperp = np.interp(wavelength*1e-3,data_perp['table']['wavelength_um'][::-1], data_perp['table']['Im_n'][::-1])
            Impar = np.interp(wavelength*1e-3,data_par['table'][::-1]['wavelength_um'], data_par['table']['Im_n'][::-1])
            Imperp = Imperp[::-1]
            Impar = Impar[::-1]
        elif grain_types[i] == 'silicate':
            data = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/eps_suvSil')
            Im = np.interp(wavelength*1e-3,data['table']['wavelength_um'][::-1], data['table']['Im_n'][::-1])
            Im = Im[::-1]

        # 5. Interpolate the absorption cross section for the grains
        if grain_types[i] == 'graphite':
            if a[i] == 0.1:
                dustname = 'CLarge'
                IM19_size = '1000A'
                linestyle = '-'
            elif a[i] == 0.005:
                dustname = 'CSmall'
                IM19_size = '50A'
                linestyle = '-.'
            a0,wav,_,C_abs,_ = compute_cross_sections(dustname,do_average=False)
        elif grain_types[i] == 'silicate':
            if a[i] == 0.1:
                dustname = 'SilLarge'
                IM19_size = '1000A'
                linestyle = '--'
            elif a[i] == 0.005:
                dustname = 'SilSmall'
                IM19_size = '50A'
                linestyle = ':'
            a0,wav,_,C_abs,_ = compute_cross_sections(dustname,do_average=False)
        a_cm = a[i] * 1e-4
        # Convert wavelength to energy in eV
        optical_E = 1.2398 / (wav*1e4)  # wav in microns, E in eV
        interp_C_abs = np.interp(E,optical_E,C_abs) # Interpolate C_abs to the photon energies

        # 6. Loop over temperatures and compute the photoelectric heating rate
        peh_rates = np.zeros(nT)
        rec_rates = np.zeros(nT)
        peh_rates_mean = np.zeros(nT)
        rec_rates_mean = np.zeros(nT)
        for j in range(0, nT):
            # 7. Compute the equilibrium grain charge distribution
            if use_equilibrium:
                # lazily import to avoid circular imports
                from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
                a_cm_eq = a[i] * 1e-4
                Zs, P, rates, Zmean, Zsigma = equilibrium_charge_for_grain(G0, ne, temperatures[j],
                                                                          grain_types[i], a_cm_eq,
                                                                          radiation_model=radiation_model,
                                                                          rad_field=None, yield_params=None,
                                                                          Z_start=0, debug=False)
                grain_charge_pdf = P
                grain_charges = Zs
            else:
                grain_charge_pdf, grain_charges = grain_charge_dist(G0,temperatures[j],ne,
                                                                    grain_types[i],IM19_size)
            args = []
            for k in range(0, len(grain_charges)):
                args.append((grain_charges[k],a[i]*1e3,rad_field,grain_types[i],
                             np.column_stack([Imperp,Impar]) if grain_types[i] == 'graphite' else Im,
                             interp_C_abs))
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                results = list(tqdm(executor.map(compute_photoelectric_heating_rate, args), total= len(grain_charges),
                                    desc=f'    Computing efficiency for {rad_name} field', unit=' steps'))
            peh_rates[j] = np.sum(np.array(results) * grain_charge_pdf)

            args = []
            for k in range(0, len(grain_charges)):
                args.append((grain_charges[k],a_cm,ne,temperatures[j],grain_types[i]))
            with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                results = list(tqdm(executor.map(compute_recombination_cooling_rate, args), total= len(grain_charges),
                                    desc=f'    Computing efficiency for {rad_name} field', unit=' steps'))
            rec_rates[j] = np.sum(np.array(results) * grain_charge_pdf)

            # 8. Compute the mean grain charge
            if use_equilibrium:
                # use distribution we just computed
                grain_mc = float(np.sum(np.asarray(grain_charges) * np.asarray(grain_charge_pdf)))
            else:
                grain_mc = grain_mean_charge(G0,temperatures[j],ne,
                                                                    grain_types[i],IM19_size)
            args = (grain_mc,a[i]*1e3,rad_field,grain_types[i],
                    np.column_stack([Imperp,Impar]) if grain_types[i] == 'graphite' else Im,
                    interp_C_abs)
            peh_rates_mean[j] = compute_photoelectric_heating_rate(args)
            args = (grain_mc,a_cm,ne,temperatures[j],grain_types[i])
            rec_rates_mean[j] = compute_recombination_cooling_rate(args)

        # 9. Plot the results for this grain type
        # Store handles for custom legends
        from matplotlib.lines import Line2D
        if i == 0:
            color_handles = [Line2D([0], [0], color='r', lw=2, label=r'$\Gamma_{\rm peh}$ (mean)'),
                            Line2D([0], [0], color='orange', lw=2, label=r'$\Gamma_{\rm peh}$ (all)'),
                            Line2D([0], [0], color='b', lw=2, label=r'$\Lambda_{\rm rec}$ (mean)'),
                            Line2D([0], [0], color='lightskyblue', lw=2, label=r'$\Lambda_{\rm rec}$ (all)')]
        ax.plot(temperatures, peh_rates_mean, color='r', linestyle=linestyle, linewidth=2)
        ax.plot(temperatures, peh_rates, color='orange', linestyle=linestyle, linewidth=2)
        ax.plot(temperatures, rec_rates_mean, color='b', linestyle=linestyle, linewidth=2)
        ax.plot(temperatures, rec_rates, color='lightskyblue', linestyle=linestyle, linewidth=2)
        # Store a dummy handle for each grain type for linestyle legend
        if 'ls_handles' not in locals():
            ls_handles = []
            ls_labels = []
        ls_handles.append(Line2D([0], [0], color='black', linestyle=linestyle, lw=2))
        ls_labels.append(fr'{grain_types[i]}, $a={a[i]}$ $\mu$m')

    # Add color legend (heating/cooling)
    legend1 = ax.legend(color_handles, [h.get_label() for h in color_handles], loc='lower left', fontsize=12, frameon=False, title='Heating/Cooling')
    # Add linestyle legend (grain type)
    legend2 = ax.legend(ls_handles, ls_labels, loc='upper right', fontsize=12, frameon=False, title='Grain type')
    ax.add_artist(legend1)
    fig.subplots_adjust(top=0.97, bottom=0.11, left=0.13, right=0.98, wspace=0, hspace=0)
    fig.savefig(_photoelectric_output_path(f'dust_photoelectric_heating_vs_recombination_{radiation_model}.pdf'), format='pdf', dpi=300)

def plot_DS87_thetanu():
    use_calima_style()
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$\theta_{\nu}/\nu$', fontsize=16)
    ax.set_xlabel(r'$\nu$', fontsize=16)
    ax.tick_params(labelsize=14)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.plot(DS87_nu,DS87_theta_nu)

    fig.savefig(_photoelectric_output_path('DS87_thetanu.pdf'), format='pdf', dpi=300)

def get_radiation_field(radiation_model, E_min=0.1, E_max=13.6):
    """
    Get the radiation field for a given radiation model.
    Parameters
    ----------
    radiation_model : str
        Radiation model to use ('Draine', 'Habing', 'Mathis', 'HD200775', 'BB{T_star}', 'O6V', 'B0V', 'A0', 'BPASS_veryyoung_lowz', 'BPASS_young_midz', 'BPASS_old_highz').
        For blackbody, use 'BB{T_star}' where {T_star} is the temperature in K (e.g., 'BB30000' for 30000 K).
    E_min : float, optional
        Minimum energy in eV (default: 0.1 eV).
    E_max : float, optional
        Maximum energy in eV (default: 13.6 eV).
    Returns
    -------
    rad_field : np.ndarray
        Radiation field as a 2D array with columns [energy (eV), wavelength (nm), intensity (erg/s/cm^2/eV)].
    """
    from astropy.table import Table
    from pycalima.models.PAH_charge.PAH_photoelectric_heating import blackbody_radiation
    from pycalima.models.tools.radiation_fields import Draine_1978_isrf
    if radiation_model == 'Draine':
        draine_data = np.loadtxt(_external_data_path('draine1978.dat'))
        wavelength_nm = np.asarray(draine_data[:, 0], dtype=float)
        photon_flux = np.asarray(Draine_1978_isrf(wavelength_nm), dtype=float)
        energy_eV = 1239.84193 / wavelength_nm
        dlam_dE_nm_per_eV = 1239.84193 / np.maximum(energy_eV ** 2, 1e-300)
        I_lambda = photon_flux * energy_eV * eV2erg
        I_E = I_lambda * dlam_dE_nm_per_eV
        rad_field = np.column_stack([energy_eV, wavelength_nm, I_E])
        rad_label = 'Draine (1978)'
    elif radiation_model == 'Habing':
        draine_data = np.loadtxt(_external_data_path('draine1978.dat'))
        wavelength_nm = np.asarray(draine_data[:, 0], dtype=float)
        photon_flux = np.asarray(Draine_1978_isrf(wavelength_nm), dtype=float) / 1.7
        energy_eV = 1239.84193 / wavelength_nm
        dlam_dE_nm_per_eV = 1239.84193 / np.maximum(energy_eV ** 2, 1e-300)
        I_lambda = photon_flux * energy_eV * eV2erg
        I_E = I_lambda * dlam_dE_nm_per_eV
        rad_field = np.column_stack([energy_eV, wavelength_nm, I_E])
        rad_label = 'Habing (1968)'
    elif radiation_model == 'Mathis':
        # Use the analytic Mathis (1983) functional form from Mathis83_radiation_field
        # Mathis83_radiation_field(E) returns u_E in units erg / cm^3 / eV.
        # To be consistent with other radiation models (which return
        # wavelength [nm] and I_lambda [erg / cm^2 / s / nm / sr]) we convert
        # the functional u_E -> I_lambda per nm per sr and return a 2-column array
        # [wavelength_nm, I_lambda].
        E = np.linspace(13.6,0.01, 2000)
        u_E = np.asarray([Mathis83_radiation_field(e) for e in E])
        # constants
        h_SI = 6.62607015e-34
        eV2J = 1.602176634e-19
        c_SI = 2.99792458e8
        c_cgs = 2.99792458e10

        # dnu/dE (Hz per eV)
        dnu_dE = eV2J / h_SI

        # u_nu (erg cm^-3 Hz^-1) = u_E / (dnu/dE)
        u_nu = u_E / dnu_dE

        # I_nu (erg cm^-2 s^-1 Hz^-1 sr^-1) = c / (4π) * u_nu  [using c in cm/s]
        I_nu = (c_cgs / (4.0 * np.pi)) * u_nu

        # convert to I_lambda: I_lambda = I_nu * c / lambda^2
        E_J = E * eV2J
        lambda_m = h_SI * c_SI / E_J
        lambda_cm = lambda_m * 100.0
        I_lambda_per_cm = I_nu * c_cgs / (lambda_cm ** 2)

        # convert per cm to per nm: 1 nm = 1e-7 cm -> I_per_nm = I_per_cm * 1e-7
        I_lambda_per_nm = I_lambda_per_cm * 1e-7

        wavelength_nm = 1239.84193 / E
        rad_field = np.column_stack([wavelength_nm, I_lambda_per_nm])
        rad_label = 'Mathis et al. (1983)'
    elif radiation_model == 'Mathis_file':
        # Explicit file-based Mathis (legacy behaviour). Returns the raw file columns
        mathis1983 = Table.read('../photoelectric-heating/ISRF/mathis1983.txt', format='ascii')
        rad_field = np.column_stack([mathis1983['col1'], mathis1983['col2']])
        rad_label = 'Mathis et al. (1983)'
    elif radiation_model == 'HD200775':
        HD200775 = Table.read('../photoelectric-heating/stars/HD200775_RF.txt', format='ascii')
        rad_field = np.column_stack([HD200775['col1'],HD200775['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_label = 'HD200775'
    elif radiation_model[:2] == 'BB':
        T_star = float(radiation_model[2:])
        # Obtain the black body radiation field in units of erg cm-2 s-1 nm-1 sr-1 for the 
        # given temperature
        BB = blackbody_radiation(T_star, 23.0, 500, num_points=1000)
        rad_field = np.column_stack([BB[0].to('nm').d,BB[1].to('erg/cm**2/s/nm').d])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_label = f'BB T={T_star} K'
    elif radiation_model == 'O6V':
        O6V = Table.read('../photoelectric-heating/stars/kp00_40000', format='ascii')
        rad_field = np.column_stack([O6V['col1'],O6V['col2']])
        distance = 20. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_label = 'O6V'
    elif radiation_model == 'B0V':
        B0V = Table.read('../photoelectric-heating/stars/kp00_30000', format='ascii')
        rad_field = np.column_stack([B0V['col1'],B0V['col2']])
        distance = 20. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_label = 'B0V'
    elif radiation_model == 'A0':
        A0 = Table.read('../photoelectric-heating/stars/kp00_10000', format='ascii')
        rad_field = np.column_stack([A0['col1'],A0['col2']])
        distance = 20. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2 
        rad_label = 'A0'
    elif radiation_model == 'BPASS_veryyoung_lowz':
        from pycalima.models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables(_bpass_sed_dir())
        fixed_age = 0.01 # 10 Myr
        fixed_metallicity = 0.0002 # 0.01 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_label = r'BPASS ($t=10$ Myr, $Z=0.01Z_{\odot}$)'
    elif radiation_model == 'BPASS_young_midz':
        from pycalima.models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables(_bpass_sed_dir())
        fixed_age = 0.1 # 0.1 Gyr
        fixed_metallicity = 0.01 # 0.5 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_label = r'BPASS ($t=0.1$ Gyr, $Z=0.5Z_{\odot}$)'
    elif radiation_model == 'BPASS_old_highz':
        from pycalima.models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables(_bpass_sed_dir())
        fixed_age = 1 # 1 Gyr
        fixed_metallicity = 0.02 # 1 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_label = r'BPASS ($t=1$ Gyr, $Z=Z_{\odot}$)'
    
    global _RADIATION_FIELD_LOGGED_ONCE

    # Filter radiation field to energy range [E_min, E_max] eV.
    # Two-column fields are wavelength-first; the Draine/Habing loader above
    # returns a three-column energy-first field already.
    if rad_field.ndim == 2 and rad_field.shape[1] >= 3:
        energy_eV = rad_field[:, 0].astype(float)
    else:
        # Convert wavelength (nm) to energy (eV): E = 1239.84193 / lambda_nm
        energy_eV = 1239.84193 / rad_field[:, 0]
    energy_mask = (energy_eV >= E_min) & (energy_eV <= E_max)
    rad_field = rad_field[energy_mask, :]

    # Print the status line only once and only in the main process.
    if not _RADIATION_FIELD_LOGGED_ONCE:
        try:
            import multiprocessing as _mp
            is_main_process = (_mp.current_process().name == 'MainProcess')
        except Exception:
            is_main_process = True
        if is_main_process:
            print(f'[get_radiation_field] Loaded radiation field: {radiation_model} with energy range [{E_min}, {E_max}] eV, {rad_field.shape[0]} points.')
            _RADIATION_FIELD_LOGGED_ONCE = True

    return rad_field,rad_label

def compute_peh_model(grain_type, radiation_model, a_cm, ne, T, n_gamma=10, sweep_variable='G0'):
    """
    Compute the photoelectric heating efficiency for a given grain type and radiation model.
    
    Parameters
    ----------
    grain_type : str
        Grain type ('graphite' or 'silicate').
    radiation_model : str
        Radiation model to use ('Draine', 'Mathis', 'HD200775', 'BB{T_star}', 'O6V', 'B0V', 'A0', 'BPASS_veryyoung_lowz', 'BPASS_young_midz', 'BPASS_old_highz').
        For blackbody, use 'BB{T_star}' where {T_star} is the temperature in K (e.g., 'BB30000' for 30000 K).
    a_cm : float
        Grain size in cm.
    T : float
        Gas temperature in K.
    ne_min : float
        Minimum electron density in cm^-3.
    ne_max : float
        Maximum electron density in cm^-3.
    n_ne : int, optional
        Number of electron density points to compute (default is 10).

    Returns
    -------
    gamma : ndarray
        Gamma parameter (G0 * sqrt(T) / ne) in K^0.5 cm^3.
    E_abs : float
        Total power absorbed by the grain in erg s^-1.
    peh_rate : ndarray
        Photoelectric heating rate in erg s^-1.
    rec_rate : ndarray
        Recombination rate in erg s^-1.
    """
    from pycalima.models.dust_radiation.dust_emission import compute_cross_sections,interpolate_cross_sections
    from unyt import nm,m,cm,eV,J,s,h,c,erg,K,kb
    import concurrent.futures
    from tqdm import tqdm

    # 1. Loop over a fixed gamma grid and compute the photoelectric heating efficiency
    # gamma is defined as G0 * sqrt(T) / ne. We always sample gamma from 1e-1 to 1e6
    gamma_grid = np.logspace(-1, 6, n_gamma)
    peh_rate = np.zeros(n_gamma)
    rec_rate = np.zeros(n_gamma)
    E_abs_arr = np.zeros(n_gamma)
    # record the actual parameters used per sample for diagnostics
    G0_used_arr = np.full(n_gamma, np.nan)
    ne_used_arr = np.full(n_gamma, np.nan)
    T_used_arr = np.full(n_gamma, np.nan)

    # 2. Pre-import equilibrium solver (always used)
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain, compute_G0_from_rad_field

    # 3. compute G0 for the asked radiation field
    rad0, _ = get_radiation_field(radiation_model)
    G0_base, power = compute_G0_from_rad_field(rad0)
    print(f'[compute_peh_model] G0 for {radiation_model}: {G0_base:.3e} (power={power:.3e})')

    # 4. Loop over gamma grid
    for i in range(0, n_gamma):
        # gamma value for this sample
        gamma_val = float(gamma_grid[i])

        # derive G0_used, ne_used, T_used depending on sweep_variable
        sv = str(sweep_variable).lower()
        if sv == 'g0':
            ne_used = float(ne)
            T_used = float(T)
            G0_used = gamma_val * ne_used / np.sqrt(T_used)
        elif sv == 'ne':
            G0_used = float(G0_base)
            T_used = float(T)
            ne_used = G0_used * np.sqrt(T_used) / gamma_val
        elif sv in ('t', 'temp', 'temperature'):
            G0_used = float(G0_base)
            ne_used = float(ne)
            T_used = (gamma_val * ne_used / G0_used) ** 2.0
        else:
            raise ValueError(f'Unknown sweep_variable: {sweep_variable}. Choose one of "G0", "ne", or "T"')

        # record parameters for diagnostics
        G0_used_arr[i] = G0_used
        ne_used_arr[i] = ne_used
        T_used_arr[i] = T_used

        # obtain equilibrium solution (always use equilibrium solver)
        Zs_eq, P_eq, rates, Zmean, Zsigma = equilibrium_charge_for_grain(
            G0_used, ne_used, T_used, grain_type, a_cm,
            radiation_model=radiation_model, rad_field=None, yield_params=None,
            Z_start=0, debug=False)


        peh_rate[i] = float(rates.get('Gamma_total', 0.0))
        rec_rate[i] = float(rates.get('Recomb_total', 0.0)) + float(rates.get('Autoionisation_cooling', 0.0))
        E_abs_arr[i] = float(rates.get('E_abs', np.nan)) if 'E_abs' in rates else np.nan


    # 5. Gamma grid used
    gamma = gamma_grid

    # print explored ranges for the trio (G0, ne, T)
    try:
        valid = np.isfinite(G0_used_arr)
        if np.any(valid):
            print(f'[compute_peh_model] Explored G0 range: {np.nanmin(G0_used_arr):.3e} -> {np.nanmax(G0_used_arr):.3e} (n={np.count_nonzero(valid)})')
            print(f'[compute_peh_model] Explored ne range: {np.nanmin(ne_used_arr):.3e} -> {np.nanmax(ne_used_arr):.3e} cm^-3')
            print(f'[compute_peh_model] Explored T range: {np.nanmin(T_used_arr):.3e} -> {np.nanmax(T_used_arr):.3e} K')
        else:
            print('[compute_peh_model] No valid points computed (all NaN)')
    except Exception:
        pass

    return gamma, E_abs_arr, peh_rate, rec_rate


def compute_tables_ISRF(ne, T, n_gamma=10, radiation_model='Mathis', sweep_variable='G0'):

    """
    Compute the photoelectric heating efficiency tables for different grain types and sizes.
    
    Parameters
    ----------
    ne : float
        Electron density in cm^-3.
    T_min : float
        Minimum temperature in K.
    T_max : float
        Maximum temperature in K.
    n_T : int, optional
        Number of temperature points to compute (default is 10).
    radiation_model : str, optional
        Radiation model to use ('Draine', 'Mathis', 'HD200775', 'BB{T_star}', 'O6V', 'B0V', 'A0', 'BPASS_veryyoung_lowz', 'BPASS_young_midz', 'BPASS_old_highz').
        For blackbody, use 'BB{T_star}' where {T_star} is the temperature in K (e.g., 'BB30000' for 30000 K).

    """
    use_calima_style()
    import os
    # 1. Define grain types and sizes
    grain_types = [('graphite',1e-6,'steelblue','Gra'),('graphite',1e-5,'cornflowerblue','Gra'),
                   ('silicate',5e-7,'saddlebrown','suvSil'),('silicate',1e-5,'sandybrown','suvSil')]
    # grain_types = [('silicate',5e-3,'saddlebrown','suvSil')]

    # 2. Setup the figure
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$\Gamma_{\rm PEH},\Lambda_{\rm rec}$ [erg s$^{-1}$]', fontsize=16)
    ax.set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim([1e-19,2e-14])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()


    # 3. Loop over grain types and compute the photoelectric heating efficiency
    for grain_type, a_cm, color, name in grain_types:
        gamma, E_abs_arr, peh_rate, rec_rate = compute_peh_model(grain_type, radiation_model, a_cm, ne, T, n_gamma=n_gamma, sweep_variable=sweep_variable)
        print('gamma:', gamma)
        print('peh_rate:', peh_rate)
        print('rec_rate:', rec_rate)
        # compute efficiency per G0 (safely handle zeros)
        efficiency = np.zeros_like(peh_rate)
        for ii in range(len(peh_rate)):
            if E_abs_arr[ii] is None or not np.isfinite(E_abs_arr[ii]) or E_abs_arr[ii] == 0.0:
                efficiency[ii] = 0.0
            else:
                efficiency[ii] = (peh_rate[ii] - rec_rate[ii]) / E_abs_arr[ii]
        ax.plot(gamma, peh_rate, label=fr'{grain_type}, $a={a_cm:.2e}$ cm', color=color, linestyle='-', linewidth=2)
        ax.plot(gamma, rec_rate, label=fr'{grain_type}, $a={a_cm:.2e}$ cm', color=color, linestyle='--', linewidth=2)

        # 4. Save the results to a file
        peh_table_dir = _photoelectric_output_path('dust_PEH_tables')
        os.makedirs(peh_table_dir, exist_ok=True)
    # Save enriched table: gamma, peh_rate, rec_rate, E_abs, efficiency
    outdata = np.column_stack([gamma, peh_rate, rec_rate, E_abs_arr, efficiency])
    header = 'gamma[ G0*sqrt(T)/ne ], peh_rate[erg/s], rec_rate[erg/s], E_abs[erg/s], efficiency'
    np.savetxt(_photoelectric_output_path(f'dust_PEH_tables/dust_PEH_{a_cm:.3e}_cm_{name}_sweep-{sweep_variable}.txt'), outdata, header=header, fmt='%14.6e', comments='')


    # 5. Add the legend for the grain types and second artist legend for line styles
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color='k', lw=2, label=r'$\Gamma_{\rm PEH}$'),
                    Line2D([0], [0], color='k', lw=2, linestyle='--', label=r'$\Lambda_{\rm rec}$')]
    legend1 = ax.legend(color_handles, [h.get_label() for h in color_handles], loc='lower left', fontsize=12, frameon=False, title='Heating/Cooling')
    # Add linestyle legend (grain type)
    ls_handles = []
    ls_labels = []
    for grain_type, a_cm, color, dustname in grain_types:
        ls_handles.append(Line2D([0], [0], color=color, linestyle='-', lw=2))
        ls_labels.append(fr'{grain_type}, $a={a_cm:.2e}$ cm')
    legend2 = ax.legend(ls_handles, ls_labels, loc='upper right', fontsize=12, frameon=False, title='Grain type')
    ax.add_artist(legend1)

    # 6. Adjust and save the figure
    fig.subplots_adjust(top=0.97, bottom=0.11, left=0.13, right=0.98, wspace=0, hspace=0)
    fig.savefig(_photoelectric_output_path(f'dust_photoelectric_heating_vs_recombination_{radiation_model}.pdf'), format='pdf', dpi=300)

def plot_efficiency(T,ne,radiation_model='Draine',G0factor=1.0,nsizes=50):
    use_calima_style()
    import concurrent.futures
    from tqdm import tqdm

    epsilon0_SI = 8.854187817e-12  # F/m
    e_SI = 1.602176634e-19         # C
    angstrom_to_cm = 1e-8
    cm_to_angstrom = 1e8

    
    # 1. Setup the figure
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,3.5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$\epsilon_{\rm PEH}$', fontsize=16)
    ax.set_xlabel(r'$a$ [$\AA$]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    fig_pot, ax_pot = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
    ax_pot.set_xscale('log')
    ax_pot.set_xlabel(r'grain size $a$ [\AA]')
    ax_pot.set_ylabel('surface potential (V)')
    ax_pot.grid(True, which='both', ls=':', alpha=0.5)
    ax_pot.set_ylim([-2,8])

    grain_types = ['graphite','silicate']
    from pycalima.models.dust_radiation.dust_emission import USE_LI_DRAINE_2001_CARBONACEOUS
    min_size_angstrom = 3 if USE_LI_DRAINE_2001_CARBONACEOUS else 10.0
    grain_sizes = np.logspace(np.log10(min_size_angstrom * angstrom_to_cm), np.log10(10000 * angstrom_to_cm), nsizes)  # in cm

    # Read the two column file 1e4_Draine.csv
    data = np.loadtxt(_external_data_path('1e4_Draine.csv'), delimiter=',')
    asize_draine = data[:, 0]  # in Angstroms
    efficiency_draine = data[:, 1]
    ax.plot(asize_draine, efficiency_draine, label=r'Weingartner \& Draine 2001 (graphite)', color='k', linestyle=':', linewidth=2)

    data = np.loadtxt(_external_data_path('1e4_silicate_draine.csv'), delimiter=',')
    asize_draine = data[:, 0]  # in Angstroms
    efficiency_draine = data[:, 1]
    ax.plot(asize_draine, efficiency_draine, label=r'Weingartner \& Draine 2001 (silicate)', color='k', linestyle='-.', linewidth=2)

    # plot the results from Draine_potential_graphite.csv and Draine_potential_silicate.csv
    for mat in grain_types:
        if mat == 'graphite':
            data = np.loadtxt(_external_data_path('Draine_potential_graphite_WNM.csv'), delimiter=',', skiprows=1)
            linestyle=':'
        elif mat == 'silicate':
            data = np.loadtxt(_external_data_path('Draine_potential_silicate_WNM.csv'), delimiter=',', skiprows=1)
            linestyle='-.'
        sizes_draine = data[:, 0]  # in Angstroms
        potentials_draine = data[:, 1]  # in eV
        ax_pot.plot(sizes_draine, potentials_draine, label=f'{mat} (Draine 2011)', linestyle=linestyle, color='k')

    for i in range(0, len(grain_types)):
        eff = np.zeros(len(grain_sizes))
        potentials = np.zeros(len(grain_sizes))
        color = 'steelblue' if grain_types[i] == 'graphite' else 'saddlebrown'
        for j in range(0, len(grain_sizes)):

            # full equilibrium (from dust_charging) — lazy import and tolerant to failures
            from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
            Zs_eq, P_eq, rates_eq, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
                G0factor, ne, T, grain_types[i], grain_sizes[j],
                radiation_model=radiation_model, rad_field=None, yield_params=None,
                Z_start=0, debug=False)

            # choose which distribution to use for downstream PEH/recombination
            potentials[j] = Zmean_eq * e_SI / (4.0 * np.pi * epsilon0_SI * grain_sizes[j] * 1e-2)

            peh_rate = float(rates_eq.get('Gamma_total', 0.0))
            rec_rate = float(rates_eq.get('Recomb_total', 0.0)) + float(rates_eq.get('Autoionisation_cooling', 0.0))
            ai_rate = float(rates_eq.get('Autoionisation_cooling', 0.0))

            E_abs = float(rates_eq.get('E_abs', np.nan)) if 'E_abs' in rates_eq else None
            eff_val = float(rates_eq.get('efficiency', np.nan)) if 'efficiency' in rates_eq else None

            eff[j] = float(eff_val)
            print(f'Grain: {grain_types[i]}, a={grain_sizes[j] * cm_to_angstrom} A, PEH rate: {peh_rate:.3e}, Rec rate: {rec_rate:.3e}, Autoionisation rate: {ai_rate:.3e}, E_abs: {E_abs:.3e}, Efficiency: {eff[j]:.3e}')

        # 5. Plot the results for this grain type
        ax.plot(grain_sizes * cm_to_angstrom, eff, label=fr'{grain_types[i]}',
                color=color, linestyle='-', linewidth=2)
        
        ax_pot.plot(grain_sizes * cm_to_angstrom, potentials, label=fr'{grain_types[i]}',
                color=color, linestyle='-', linewidth=2)
    
    # 6. Add legend and savefig
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    ax_pot.legend(loc='upper right', fontsize=12, frameon=False)
    
    fig.subplots_adjust(top=0.97, bottom=0.15, left=0.11, right=0.98, wspace=0, hspace=0)
    fig.savefig(_photoelectric_output_path(f'dust_photoelectric_heating_efficiency_{radiation_model}.pdf'), format='pdf', dpi=300)

    fig_pot.subplots_adjust(top=0.97, bottom=0.11, left=0.13, right=0.98, wspace=0, hspace=0)
    fig_pot.savefig(_photoelectric_output_path(f'dust_photoelectric_heating_potential_{radiation_model}.pdf'), format='pdf', dpi=300)

def _compute_efficiency_for_field(task):
    """Compute efficiency arrays for all grain_types for one radiation field (for multiprocessing)."""
    rad, grain_types, grain_sizes, T, ne, G0factor = task
    field_result = []
    for grain_type in grain_types:
        eff = np.zeros(len(grain_sizes))
        for j, a_cm in enumerate(grain_sizes):
            from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
            Zs_eq, P_eq, rates_eq, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
                G0factor, ne, T, grain_type, a_cm,
                radiation_model=rad, rad_field=None, yield_params=None,
                Z_start=0, debug=False)
            eff_val = float(rates_eq.get('efficiency', np.nan)) if 'efficiency' in rates_eq else np.nan
            eff[j] = eff_val
        field_result.append((grain_type, eff))
    return rad, field_result

def plot_efficiency_all_fields(T, ne, G0factor=1.0, nsizes=50,
                               grain_types=None, radiation_models=None):
    """
    Recompute PEH efficiency for graphite and silicate across the radiation
    fields used in compare_eff_curves_ISRF, using colors for radiation fields
    and linestyles for grain types so both are visually distinguished.
    """
    use_calima_style()

    if grain_types is None:
        grain_types = ['graphite', 'silicate']
    if radiation_models is None:
        radiation_models = [
            'Draine', 'Mathis', 'Habing', 'O6V', 'A0',
            'BPASS_veryyoung_lowz', 'BPASS_old_highz'
        ]
        # radiation_models = [
        #     'O6V', 'Mathis'
        # ]

    rad_colors = {
        'Draine': '#BBD8B3',
        'Habing': '#F3B61F',
        'Mathis': '#6A994E',
        'O6V': '#C33149',
        'B0V': '#4B543B',
        'A0': '#533A71',
        'BPASS_veryyoung_lowz': '#258EA6',
        'BPASS_young_midz': '#F75590',
        'BPASS_old_highz': '#D84A05'
    }
    grain_styles = {
        'graphite': '-',
        'silicate': '--'
    }
    angstrom_to_cm = 1e-8
    cm_to_angstrom = 1e8

    # Plot all radiation field SEDs for inspection
    fig_sed, ax_sed = plt.subplots(1, 1, figsize=(8, 5), dpi=300, facecolor='w', edgecolor='k')
    ax_sed.set_ylabel(r'Intensity [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$ sr$^{-1}$]', fontsize=14)
    ax_sed.set_xlabel(r'Wavelength [nm]', fontsize=14)
    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')
    ax_sed.tick_params(labelsize=12, which='both', direction='in')
    ax_sed.xaxis.set_ticks_position('both')
    ax_sed.yaxis.set_ticks_position('both')
    ax_sed.minorticks_on()

    rad_labels = {}
    for rad in radiation_models:
        rad_field, rad_label = get_radiation_field(rad)
        rad_labels[rad] = rad_label
        color = rad_colors.get(rad, '#000000')
        if rad_field.shape[1] >= 3:
            ax_sed.plot(rad_field[:, 1], rad_field[:, 2], label=rad_label, color=color, linewidth=2)
        else:
            ax_sed.plot(rad_field[:, 0], rad_field[:, 1], label=rad_label, color=color, linewidth=2)

    ax_sed.legend(loc='best', fontsize=12, frameon=False)
    fig_sed.subplots_adjust(top=0.95, bottom=0.12, left=0.12, right=0.97)
    fig_sed.savefig(_photoelectric_output_path(f'radiation_fields_comparison_T{int(T)}_ne{ne:.1e}.pdf'), format='pdf', dpi=300)

    from pycalima.models.dust_radiation.dust_emission import USE_LI_DRAINE_2001_CARBONACEOUS
    min_size_angstrom = 4.0 if USE_LI_DRAINE_2001_CARBONACEOUS else 10.0
    grain_sizes = np.logspace(np.log10(min_size_angstrom * angstrom_to_cm), np.log10(10000 * angstrom_to_cm), nsizes)  # cm

    # Parallelise by radiation field
    import concurrent.futures

    tasks = [(rad, grain_types, grain_sizes, T, ne, G0factor) for rad in radiation_models]
    max_workers = min(len(tasks), os.cpu_count() or 1)
    eff_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rad, field_result in executor.map(_compute_efficiency_for_field, tasks):
            eff_results[rad] = {gt: eff for gt, eff in field_result}

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$\epsilon_{\rm PEH}$', fontsize=15)
    ax.set_xlabel(r'$a$ [$\AA$]', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=12, which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.set_xlim([10, 1e4])

    # Read the two column file 1e4_Draine.csv
    data = np.loadtxt(_external_data_path('1e4_Draine.csv'), delimiter=',')
    asize_draine = data[:, 0]  # in Angstroms
    efficiency_draine = data[:, 1]
    ax.plot(asize_draine, efficiency_draine, color='k', linestyle='-', linewidth=2)

    data = np.loadtxt(_external_data_path('1e4_silicate_draine.csv'), delimiter=',')
    asize_draine = data[:, 0]  # in Angstroms
    efficiency_draine = data[:, 1]
    ax.plot(asize_draine, efficiency_draine, color='k', linestyle='--', linewidth=2)

    for grain_type in grain_types:
        linestyle = grain_styles.get(grain_type, '-')
        for rad in radiation_models:
            eff = eff_results.get(rad, {}).get(grain_type, np.full(len(grain_sizes), np.nan))
            print(eff)
            # Separate heating (positive) and cooling (negative) efficiencies
            eff_heating = np.where((eff > 0) & np.isfinite(eff), eff, np.nan)
            eff_cooling = np.where((eff < 0) & np.isfinite(eff), np.abs(eff), np.nan)
            
            color = rad_colors.get(rad, '#000000')
            # Plot heating with full opacity
            ax.plot(grain_sizes * cm_to_angstrom, eff_heating, label=f'{rad} | {grain_type}',
                    color=color, linestyle=linestyle, linewidth=2, alpha=1.0)
            # Plot cooling (absolute value) with lower opacity
            ax.plot(grain_sizes * cm_to_angstrom, eff_cooling,
                    color=color, linestyle=linestyle, linewidth=2, alpha=0.3)

    # Build decoupled legends: colors for radiation, linestyles for grain type
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=c, lw=2) for c in rad_colors.values()]
    color_labels = rad_labels.values()
    style_handles = [Line2D([0], [0], color='k', lw=2, linestyle=ls)
                     for ls in grain_styles.values()]
    style_labels = list(grain_styles.keys())

    legend1 = ax.legend(color_handles, color_labels, title='Radiation field',
                        loc='lower left', fontsize=10, frameon=False,ncol=2)
    legend2 = ax.legend(style_handles, style_labels, title='Grain type',
                        loc='lower right', fontsize=10, frameon=False)
    ax.add_artist(legend1)

    fig.subplots_adjust(top=0.99, bottom=0.1, left=0.08, right=0.98)
    fig.savefig(_photoelectric_output_path(f'dust_photoelectric_heating_efficiency_multiISRF_T{int(T)}_ne{ne:.1e}.pdf'),
                format='pdf', dpi=300)

def Planck_function(T, nu):
    """
    Compute the Planck function for a given temperature and frequency.
    
    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    nu : float or array-like
        Frequency in Hz.
        
    Returns
    -------
    B_nu : float or array-like
        Planck function in erg cm^-2 s^-1 Hz^-1 sr^-1.
    """
    # Numerical Planck function: return B_nu in erg cm^-2 s^-1 Hz^-1 sr^-1
    # Accept scalar or array nu in Hz and T in K
    h_SI = 6.62607015e-34
    kB_SI = 1.380649e-23
    c_SI = 2.99792458e8

    nu = np.asarray(nu, dtype=float)
    T = float(T)

    # B_nu in SI: W m^-2 Hz^-1 sr^-1
    with np.errstate(over='ignore'):
        exponent = np.exp(h_SI * nu / (kB_SI * T))
        B_nu_SI = (2.0 * h_SI * nu ** 3) / (c_SI ** 2) / (exponent - 1.0)

    # Convert W m^-2 -> erg s^-1 cm^-2 : 1 W/m^2 = 1e3 erg s^-1 cm^-2
    B_nu_cgs = B_nu_SI * 1e3
    return B_nu_cgs


def _compute_rates_point(task):
    """Helper for parallel execution: task is a tuple (G0_used, ne_used, T_used, grain_type, a_cm, radiation_model, ion_species).
    Returns (peh, rec, Zmean, Zsigma, ion_recomb_rates, ion_recomb_rate_coefficients) (or nan/empty on error).
    """
    G0_used, ne_used, T_used, grain_type, a_cm, radiation_model = task[:6]
    ion_species = task[6] if len(task) > 6 else []

    from pycalima.models.dust_charge import dust_charging as _dc

    # Per-process cache to avoid rebuilding radiation/optical/yield setup for
    # every grid point. This is the same invariant context used in gamma scans.
    global _DPEH_WORKER_PREPARED_CONTEXTS
    try:
        _DPEH_WORKER_PREPARED_CONTEXTS
    except NameError:
        _DPEH_WORKER_PREPARED_CONTEXTS = {}

    ctx_key = (str(grain_type), float(a_cm), str(radiation_model))
    ctx = _DPEH_WORKER_PREPARED_CONTEXTS.get(ctx_key)
    if ctx is None:
        scan_ctx = _dc._prepare_gamma_scan_context(
            grain_type, a_cm, radiation_model=radiation_model, yield_params=None
        )
        ctx = {
            'nu': np.asarray(scan_ctx['nu'], dtype=float),
            'J_nu_base': np.asarray(scan_ctx['J_nu'], dtype=float),
            'C_abs_nu': np.asarray(scan_ctx['C_abs_nu'], dtype=float),
            'yield_func': scan_ctx['yield_func'],
            'yield_params': dict(scan_ctx['yield_params']),
        }
        _DPEH_WORKER_PREPARED_CONTEXTS[ctx_key] = ctx

    J_nu_scaled = ctx['J_nu_base'] * float(G0_used)

    Zs, P, rates, Zmean, Zsigma = _dc.compute_equilibrium_charge_distribution_vectorized(
        float(a_cm), float(ne_used), float(T_used), ion_species,
        ctx['nu'], J_nu_scaled, ctx['C_abs_nu'],
        yield_func=ctx['yield_func'],
        yield_params=ctx['yield_params'],
        Z_start=0,
        debug=False,
    )
    peh = float(rates.get('Gamma_total', np.nan) - float(rates.get('Autoionisation_cooling', 0.0)))
    rec = float(rates.get('Recomb_total', 0.0))
    ion_recomb_rates = rates.get('ion_recomb_rates', np.array([]))
    ion_recomb_rate_coefficients = rates.get('ion_recomb_rate_coefficients', np.array([]))
    return peh, rec, Zmean, Zsigma, ion_recomb_rates, ion_recomb_rate_coefficients


def _compute_rates_batch(batch_tasks):
    """Worker helper: compute a batch of points to reduce process-pool overhead."""

    out = []
    for task in batch_tasks:
        pos, iT, ig, G0, ne, T_task, grain_type, a_cm, radiation_model = task[:9]
        ion_species = task[9] if len(task) > 9 else []
        try:
            peh, rec, Zm, Zs, ion_rec, ion_coeff = _compute_rates_point((G0, ne, T_task, grain_type, a_cm, radiation_model, ion_species))
        except Exception as exc:
            msg = (
                '[make_rate_gamma_T_tables] Worker batch task failed: '
                f'pos={pos}, iT={iT}, ig={ig}, T={T_task:.6e} K, '
                f'G0={G0:.6e}, ne={ne:.6e} cm^-3, '
                f'grain={grain_type}, a_cm={a_cm:.3e}, radiation_model={radiation_model}'
            )
            # raise standard error with context
            raise RuntimeError(msg) from exc
        out.append((pos, peh, rec, Zm, Zs, ion_rec, ion_coeff))
    return out


def make_rate_gamma_T_tables(grain_type, a_cm, radiation_model='Mathis',
                             mode='fix_G0', fixed_value=1.0,
                             Tmin=10.0, Tmax=1e5, nT=50,
                             gamma_min=1e-6, gamma_max=1e6, n_gamma=100,
                             num_workers=None, out_dir='tables', debug=False,
                             grain_label=None, executor=None, ion_species=None):
    """
    Compute grids of photoelectric heating and recombination cooling on a
    log(T) x log(gamma) grid and write ASCII tables suitable for Fortran
    linear interpolation in log-log space.

    Returns a dict with arrays and the out_dir path.
    """
    use_calima_style()
    import os
    out_dir = _photoelectric_output_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # build grids in log space
    T_vals = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    gamma_vals = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)

    # prepare tasks in same order: rows over T, columns over gamma
    tasks = []  # (iT, ig, G0, ne, T)
    for iT, T in enumerate(T_vals):
        sqrtT = np.sqrt(T)
        for ig, gamma in enumerate(gamma_vals):
            if mode == 'fix_G0':
                G0 = float(fixed_value)
                ne = max(1e-20, (G0 * sqrtT) / float(gamma))
            elif mode == 'fix_ne':
                ne = float(fixed_value)
                G0 = max(1e-20, (float(gamma) * ne) / sqrtT)
            else:
                raise ValueError('mode must be "fix_G0" or "fix_ne"')
            tasks.append((iT, ig, G0, ne, float(T)))

    N = len(tasks)
    G0_vals = np.full(N, np.nan)
    ne_vals = np.full(N, np.nan)
    peh_vals = np.full(N, np.nan)
    rec_vals = np.full(N, np.nan)
    Zmean_vals = np.full(N, np.nan)
    Zsigma_vals = np.full(N, np.nan)
    for pos, t in enumerate(tasks):
        G0_vals[pos] = t[2]
        ne_vals[pos] = t[3]

    # Optional ion recombination lists to accumulate results per task
    n_ions = len(ion_species) if ion_species else 0
    ion_recomb_rates_vals = [None] * N
    ion_recomb_rate_coeffs_vals = [None] * N

    import concurrent.futures
    # Debug mode: run in main process for full local traceback visibility.
    if num_workers == 1:
        count = 0
        for pos, t in enumerate(tasks):
            try:
                peh, rec, Zm, Zs, ion_rec, ion_coeff = _compute_rates_point((t[2], t[3], t[4], grain_type, a_cm, radiation_model, ion_species))
            except Exception as exc:
                iT, ig, G0, ne, T_task = t
                gamma_task = gamma_vals[ig]
                msg = (
                    '[make_rate_gamma_T_tables] In-process task failed: '
                    f'pos={pos}, iT={iT}, ig={ig}, T={T_task:.6e} K, '
                    f'gamma={gamma_task:.6e}, G0={G0:.6e}, ne={ne:.6e} cm^-3, '
                    f'grain={grain_type}, a_cm={a_cm:.3e}, radiation_model={radiation_model}'
                )
                raise RuntimeError(msg) from exc
            peh_vals[pos] = peh
            rec_vals[pos] = rec
            Zmean_vals[pos] = Zm
            Zsigma_vals[pos] = Zs
            ion_recomb_rates_vals[pos] = ion_rec
            ion_recomb_rate_coeffs_vals[pos] = ion_coeff
            count += 1
            if count % 100 == 0 or count == len(tasks):
                print(f'[make_rate_gamma_T_tables] Processed {count}/{len(tasks)} tasks (in-process)')
    else:
        # Execute in parallel with batched tasks to reduce inter-process overhead.
        if num_workers is None or int(num_workers) <= 0:
            num_workers = os.cpu_count() or 1

        worker_inputs = []
        for pos, t in enumerate(tasks):
            iT, ig, G0, ne, T_task = t
            worker_inputs.append((pos, iT, ig, G0, ne, T_task, grain_type, a_cm, radiation_model, ion_species))

        # Heuristic: create multiple batches per worker while keeping enough
        # work per batch to amortize process communication.
        target_batches = max(4 * int(num_workers), 1)
        batch_size = max(16, int(np.ceil(len(worker_inputs) / float(target_batches))))
        batches = [worker_inputs[i:i + batch_size] for i in range(0, len(worker_inputs), batch_size)]

        exe = executor
        owns_executor = exe is None
        if exe is None:
            exe = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)

        try:
            # progress bar if available
            try:
                from tqdm import tqdm
                iterator = tqdm(exe.map(_compute_rates_batch, batches, chunksize=1),
                                total=len(batches), desc='Computing rates')
                use_tqdm = True
            except Exception:
                iterator = exe.map(_compute_rates_batch, batches, chunksize=1)
                use_tqdm = False

            count = 0
            for batch_out in iterator:
                for pos, peh, rec, Zm, Zs, ion_rec, ion_coeff in batch_out:
                    peh_vals[pos] = peh
                    rec_vals[pos] = rec
                    Zmean_vals[pos] = Zm
                    Zsigma_vals[pos] = Zs
                    ion_recomb_rates_vals[pos] = ion_rec
                    ion_recomb_rate_coeffs_vals[pos] = ion_coeff
                count += len(batch_out)
                if not use_tqdm and (count % 100 == 0 or count == len(worker_inputs)):
                    print(f'[make_rate_gamma_T_tables] Processed {count}/{len(worker_inputs)} tasks')
        finally:
            if owns_executor:
                exe.shutdown(wait=True)

    # reshape into (nT, n_gamma)
    G0_mat = G0_vals.reshape((nT, n_gamma))
    ne_mat = ne_vals.reshape((nT, n_gamma))
    peh_mat = peh_vals.reshape((nT, n_gamma))
    rec_mat = rec_vals.reshape((nT, n_gamma))
    Zmean_mat = Zmean_vals.reshape((nT, n_gamma))
    Zsigma_mat = Zsigma_vals.reshape((nT, n_gamma))

    if n_ions > 0:
        ion_recomb_rates_mat = np.array([r if r is not None else np.zeros(n_ions) for r in ion_recomb_rates_vals]).reshape((nT, n_gamma, n_ions))
        ion_recomb_rate_coefficients_mat = np.array([c if c is not None else np.zeros(n_ions) for c in ion_recomb_rate_coeffs_vals]).reshape((nT, n_gamma, n_ions))
    else:
        ion_recomb_rates_mat = None
        ion_recomb_rate_coefficients_mat = None

    # Table-convention fix (always applied): decompose the signed
    # recombination channel into cooling (positive part) and heating
    # (negative part) so saved cooling is non-negative and heating is smooth.
    peh_save = np.array(peh_mat, copy=True)
    rec_save = np.array(rec_mat, copy=True)
    rec_neg = np.where(np.isfinite(rec_save), np.minimum(rec_save, 0.0), 0.0)
    peh_save = peh_save - rec_neg
    rec_save = np.where(np.isfinite(rec_save), np.maximum(rec_save, 0.0), np.nan)

    # convert to log10 and replace invalid values
    fill_bad = -1e30
    with np.errstate(divide='ignore', invalid='ignore'):
        log_peh = np.log10(np.where(peh_save > 0.0, peh_save, np.nan))
        log_rec = np.log10(np.where(rec_save > 0.0, rec_save, np.nan))
    log_peh[~np.isfinite(log_peh)] = fill_bad
    log_rec[~np.isfinite(log_rec)] = fill_bad

    # Write files with grain label/bin id as the primary identifier.
    size_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
    fn_grid, fn_heating, fn_cooling = _write_photoelectric_legacy_tables(
        out_dir=out_dir,
        mode=mode,
        size_tag=size_tag,
        T_vals=T_vals,
        gamma_vals=gamma_vals,
        peh_log=log_peh,
        rec_log=log_rec,
    )

    # plot rates vs gamma for every temperature in the grid
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=200, facecolor='w', edgecolor='k')
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.log10(T_vals.min()), vmax=np.log10(T_vals.max()))

    for iT, T in enumerate(T_vals):
        color = cmap(norm(np.log10(T)))
        ax.plot(gamma_vals, np.where(peh_save[iT, :] > 0.0, peh_save[iT, :], np.nan),
                color=color, linewidth=1.5, linestyle='-')
        ax.plot(gamma_vals, np.where(rec_save[iT, :] > 0.0, rec_save[iT, :], np.nan),
                color=color, linewidth=1.5, linestyle='--')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\gamma = G_0\sqrt{T}/n_e$ [K$^{1/2}$ cm$^3$]')
    ax.set_ylabel(r'Rate [erg s$^{-1}$]')
    ax.tick_params(which='both', axis='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='k', linestyle='-', lw=1.8, label='Heating'),
        Line2D([0], [0], color='k', linestyle='--', lw=1.8, label='Recombination'),
    ]
    ax.legend(handles=style_handles, loc='lower left', fontsize=11, frameon=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10}(T/\mathrm{K})$')

    title_grain = grain_label if grain_label is not None else f'a={a_cm:.3e} cm'
    fig.suptitle(f'{grain_type}, {title_grain}, {radiation_model}, mode={mode}, fixed={fixed_value:.3e}')
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.12, right=0.92)
    fn_plot = os.path.join(out_dir, f'dust_rates_vs_gamma_by_temperature_{mode}_{size_tag}.pdf')
    fig.savefig(fn_plot, dpi=200)
    plt.close(fig)

    # write README
    readme_path = os.path.join(out_dir, 'README.md')
    with open(readme_path, 'w') as fh:
        fh.write('# Dust rate tables\n')
        fh.write('\n')
        fh.write('Files:\n')
        if fn_grid is not None:
            fh.write(f'- {os.path.basename(fn_grid)} : shared log10(T) and log10(gamma) grid (2 columns)\n')
        else:
            fh.write(f'- {os.path.basename(fn_heating)} / {os.path.basename(fn_cooling)} : embed log10(T) and log10(gamma) axes directly (see description)\n')
        fh.write(f'- {os.path.basename(fn_heating)} : n_gamma rows x nT columns, log10(heating [erg s^-1])\n')
        fh.write(f'- {os.path.basename(fn_cooling)} : n_gamma rows x nT columns, log10(cooling [erg s^-1])\n')
        fh.write('\n')
        fh.write('Rows correspond to increasing gamma (from gamma_min to gamma_max). Columns correspond to increasing T (from Tmin to Tmax).\n')
        fh.write('The signed recombination channel is always decomposed before writing tables: negative values are transferred to heating and cooling is saved as the non-negative part.\n')
        fh.write('Missing/invalid values are encoded as -1e30. Tables are plain whitespace-separated ASCII suitable for Fortran reading.\n')

    if debug:
        print(f'[make_rate_gamma_T_tables] Wrote tables to {out_dir}: shapes T={nT}, gamma={n_gamma}')

    return {
        'T_vals': T_vals,
        'gamma_vals': gamma_vals,
        'G0_vals': G0_mat,
        'ne_vals': ne_mat,
        'log_peh': log_peh,
        'log_rec': log_rec,
        'Zmean': Zmean_mat,
        'Zsigma': Zsigma_mat,
        'ion_recomb_rates_grid': ion_recomb_rates_mat,
        'ion_recomb_rate_coefficients_grid': ion_recomb_rate_coefficients_mat,
        'out_dir': os.path.abspath(out_dir)
    }


def plot_heating_cooling_surfaces(grain_type, a_cm, radiation_model='Mathis', combination='G0_vs_sqrtT_over_ne',
                                  n_x=60, n_y=60, base_ne=1.0, base_T=100.0, base_G0=None, save_prefix=None,
                                  grain_label=None):
    """
    Compute and plot 2D surfaces of photoelectric heating and recombination cooling.

    combination choices:
      - 'G0_vs_sqrtT_over_ne': x axis = G0, y axis = sqrt(T)/ne. We fix ne=base_ne and compute T=(y*ne)^2.
      - 'G0_over_ne_vs_sqrtT': x axis = G0/ne (ratio), y axis = sqrt(T). We fix ne=base_ne and compute G0 = x*ne, T = y^2.
      - 'G0sqrtT_vs_ne': x axis = G0*sqrt(T), y axis = ne. We fix T=base_T and compute G0 = x / sqrt(T).

    Parameters
    ----------
    grain_type : str
        'graphite' or 'silicate'
    a_cm : float
        grain radius in cm
    radiation_model : str
        passed to get_radiation_field
    combination : str
        one of the combinations above
    n_x, n_y : int
        grid resolution
    base_ne, base_T, base_G0 : floats
        base values used to infer the missing variable; if base_G0 is None it will be computed from the radiation model
    save_prefix : str or None
        prefix for output filenames
    """
    import time
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain, compute_G0_from_rad_field

    # compute G0_base if needed
    if base_G0 is None:
        rad0, rad_label = get_radiation_field(radiation_model)
        G0_base, _ = compute_G0_from_rad_field(rad0)
    else:
        G0_base = float(base_G0)

    # define default grids depending on combination
    if combination == 'G0_vs_sqrtT_over_ne':
        x_vals = np.logspace(-3, 6, n_x)  # G0
        y_vals = np.logspace(-4, 4, n_y)  # sqrt(T)/ne
        Xlabel = 'G0'
        Ylabel = r'$\/sqrt{T} / n_e$ (sqrt(T)/ne)'
    elif combination == 'G0_over_ne_vs_sqrtT':
        x_vals = np.logspace(-4, 4, n_x)  # G0/ne
        y_vals = np.logspace(0, 2, n_y)   # sqrt(T) from 1 to 100 (T=1..1e4)
        Xlabel = 'G0 / ne'
        Ylabel = r'$\sqrt{T}$ [K^{1/2}]'
    elif combination == 'G0sqrtT_vs_ne':
        x_vals = np.logspace(-3, 6, n_x)  # G0*sqrt(T)
        y_vals = np.logspace(-4, 2, n_y)  # ne
        Xlabel = 'G0 * sqrt(T)'
        Ylabel = r'$n_e$ [cm$^{-3}$]'
    else:
        raise ValueError('Unknown combination')

    peh_grid = np.full((n_y, n_x), np.nan)
    rec_grid = np.full((n_y, n_x), np.nan)

    t0 = time.time()
    total = n_x * n_y
    # build tasks list
    tasks = []
    index_map = []
    for j, y in enumerate(y_vals):
        for i, x in enumerate(x_vals):
            # map (x,y) to (G0_used, ne_used, T_used)
            if combination == 'G0_vs_sqrtT_over_ne':
                G0_used = float(x)
                ne_used = float(base_ne)
                T_used = (y * ne_used) ** 2.0
            elif combination == 'G0_over_ne_vs_sqrtT':
                ratio = float(x)
                ne_used = float(base_ne)
                G0_used = ratio * ne_used
                T_used = float(y) ** 2.0
            elif combination == 'G0sqrtT_vs_ne':
                ne_used = float(y)
                sqrtT = np.sqrt(float(base_T))
                G0_used = float(x) / sqrtT
                T_used = float(base_T)

            tasks.append((G0_used, ne_used, T_used, grain_type, a_cm, radiation_model))
            index_map.append((j, i))

    # execute tasks in parallel with progress bar
    import concurrent.futures
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_idx = {executor.submit(_compute_rates_point, task): idx for task, idx in zip(tasks, index_map)}
        if tqdm is not None:
            iterator = tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc='Computing grid')
        else:
            iterator = concurrent.futures.as_completed(future_to_idx)

        for future in iterator:
            j, i = future_to_idx[future]
            peh, rec = future.result()
            peh_grid[j, i] = peh
            rec_grid[j, i] = rec

    # plotting
    import matplotlib.pyplot as _plt
    fig, axes = _plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].pcolormesh(x_vals, y_vals, np.log10(np.abs(peh_grid) + 1e-40), shading='auto', cmap='viridis')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(Xlabel)
    axes[0].set_ylabel(Ylabel)
    axes[0].set_title('Photoelectric Heating')
    fig.colorbar(im0, ax=axes[0], label=r'$\log_{10}(\Gamma_{\rm PEH}/[\rm erg\/s])$')

    im1 = axes[1].pcolormesh(x_vals, y_vals, np.log10(np.abs(rec_grid) + 1e-40), shading='auto', cmap='magma')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(Xlabel)
    axes[1].set_ylabel(Ylabel)
    axes[1].set_title('Recombination+auto cooling')
    fig.colorbar(im1, ax=axes[1], label=r'$\log_{10}(\Lambda_{\rm rec}/[\rm erg\/s])$')

    grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
    prefix = save_prefix or f'heating_cooling_{grain_type}_{grain_tag}_{combination}'
    outname = prefix + '.pdf'
    outname = _photoelectric_output_path(outname)
    fig.savefig(outname, dpi=200)
    _plt.close(fig)
    print(f'Saved surface plots to {outname} (computed {total} points in {time.time()-t0:.1f}s)')
    return peh_grid, rec_grid, x_vals, y_vals


def _compute_Zmean_for_size(task):
    """Worker helper to compute Zmean for a single (G0factor, ne, T, mat, aA, radiation_model).

    This helper is defined at module scope so it can be pickled by the
    ProcessPoolExecutor. It returns either a float Zmean or None on error.
    """
    G0factor, ne, T, mat, aA, radiation_model, ion_species = task
    a_cm = float(aA) * 1e-8
    # import here so child processes import the module lazily
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
    Zs, P, rates, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
        G0factor, ne, T, mat, a_cm, ion_species=ion_species,
        radiation_model=radiation_model, rad_field=None, yield_params=None,
        debug=False)
    if Zs is not None and P is not None and len(Zs) and len(P):
        return float(np.sum(np.asarray(Zs) * np.asarray(P)))
    return None


def plot_average_potential(radiation_model='Mathis', G0factor=1.0, ne=1.0, T=100.0,
                           sizes_A=None, n_sizes=10, savefile=None):
    """
    Plot the average electrostatic potential (surface potential in Volts) as a function
    of grain size (Angstrom) for graphite and silicate materials in a given radiation field.

    Parameters
    ----------
    radiation_model : str
        Passed to `get_radiation_field` to build the radiation field used to compute G0.
    G0factor : float
        Multiplicative factor applied to the radiation field's G0.
    ne : float
        Electron density (cm^-3).
    T : float
        Temperature (K).
    sizes_A : array-like or None
        List/array of grain sizes in Angstrom. If None, a log-spaced array between 3.5 and 1e4 A is used.
    n_sizes : int
        Number of sizes to sample when sizes_A is None.
    savefile : str or None
        If provided, save the figure to this path. Otherwise saves to
        `avg_potential_{radiation_model}.pdf` in the working directory.
    """
    use_calima_style()
    # build sizes array
    if sizes_A is None:
        sizes_A = np.logspace(np.log10(10), np.log10(1e4), n_sizes)
    else:
        sizes_A = np.asarray(sizes_A, dtype=float)

    # get G0 from the radiation model
    rad, rad_label = get_radiation_field(radiation_model)
    from pycalima.models.dust_charge.dust_charging import compute_G0_from_rad_field
    G0_base, _ = compute_G0_from_rad_field(rad)

    # prepare plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=150)
    ax.set_xscale('log')
    ax.set_xlabel(r'$a$ [\AA]',fontsize=16)
    ax.set_ylabel(r'$\langle U \rangle $(V)',fontsize=16)
    # ax.set_ylim([-1.5,4.3])
    ax.set_ylim([-0.5,1.2])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in",labelsize=16)

    materials = ['graphite', 'silicate']
    linestyles = ['-', '--']

    # constants (SI)
    epsilon0_SI = 8.854187817e-12  # F/m
    e_SI = 1.602176634e-19         # C

    # plot the results from Draine_potential_graphite.csv and Draine_potential_silicate.csv
    for mat_idx, mat in enumerate(materials):
        if mat == 'graphite':
            data = np.loadtxt('Draine_potential_graphite_CNM.csv', delimiter=',', skiprows=1)
            linestyle=':'
        elif mat == 'silicate':
            data = np.loadtxt('Draine_potential_silicate_CNM.csv', delimiter=',', skiprows=1)
            linestyle='-.'
        sizes_draine = data[:, 0]  # in Angstroms
        potentials_draine = data[:, 1]  # in eV
        if mat_idx == 0:
            ax.plot(sizes_draine, potentials_draine, label=r'Weingartner \& Draine 2001', linestyle=linestyles[mat_idx], color='k',lw=2)
        else:
            ax.plot(sizes_draine, potentials_draine, linestyle=linestyles[mat_idx], color='k',lw=2)

    # fallback labels for IM19 means
    fallback_labels = np.array(['3.5A','5A','10A','50A','100A','500A','1000A'])
    fallback_sizes = np.array([3.5,5,10,50,100,500,1000])
    
    for mat_idx, mat in enumerate(materials):
        Zmean_im19 = np.zeros(len(fallback_labels))
        for idx, aA in enumerate(fallback_sizes):
            a_m = aA * 1e-10
            Zmean_im19[idx] = grain_mean_charge(G0_base, T, ne, mat, fallback_labels[idx])
        phi_im19 = Zmean_im19 * e_SI / (4.0 * np.pi * epsilon0_SI * fallback_sizes*1e-10)  # Angstrom -> m
        if mat_idx == 0:
            ax.plot(fallback_sizes, phi_im19, label=f'Ibáñez-Mejía et al. 2019', linestyle=linestyles[mat_idx], color='darkorange',lw=2)
        else:
            ax.plot(fallback_sizes, phi_im19, linestyle=linestyles[mat_idx], color='darkorange',lw=2)

    ion_species = [
        {'n': 3e-3, 'T': T, 'm': 1.6726219e-27, 'z': 1},  # H+
        {'n': 0.0042, 'T': T, 'm': 12.0 * 1.66053906660e-27, 'z': 1},  # C+
    ]

    for mat_idx, mat in enumerate(materials):
        # Parallelize the per-size equilibrium-charge calculation. We compute
        # Zmean for each grain size in parallel using a process pool and then
        # convert to surface potential. If parallel execution fails we fall
        # back to the original sequential loop.
        potentials_eq = []
        try:
            import concurrent.futures
            import os
            try:
                from tqdm import tqdm
            except Exception:
                tqdm = None

            tasks = [(G0factor, ne, T, mat, float(aA), radiation_model, ion_species) for aA in sizes_A]
            max_workers = min(len(tasks), (os.cpu_count() or 1))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # executor.map preserves order which keeps sizes_A aligned with results
                results_iter = executor.map(_compute_Zmean_for_size, tasks)
                if tqdm is not None:
                    results = list(tqdm(results_iter, total=len(tasks), desc=f'Computing Zmean ({mat})'))
                else:
                    results = list(results_iter)

            # convert Zmean -> phi for each size
            for idx, Zmean_eq_val in enumerate(results):
                aA = float(sizes_A[idx])
                a_m = aA * 1e-10
                if Zmean_eq_val is not None:
                    phi_eq = Zmean_eq_val * e_SI / (4.0 * np.pi * epsilon0_SI * a_m) if a_m > 0 else 0.0
                else:
                    phi_eq = np.nan
                potentials_eq.append(phi_eq)

        except Exception:
            # Fallback: sequential computation
            for aA in sizes_A:
                a_cm = aA * 1e-8
                a_m = aA * 1e-10
                from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
                Zs, P, rates, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
                    G0factor, ne, T, mat, a_cm, ion_species=ion_species,
                    radiation_model=radiation_model, rad_field=None, yield_params=None,
                    debug=False)
                if Zs is not None and P is not None and len(Zs) and len(P):
                    Zmean_eq_val = float(np.sum(np.asarray(Zs) * np.asarray(P)))
                else:
                    Zmean_eq_val = None

                if Zmean_eq_val is not None:
                    phi_eq = Zmean_eq_val * e_SI / (4.0 * np.pi * epsilon0_SI * a_m) if a_m > 0 else 0.0
                else:
                    phi_eq = np.nan
                potentials_eq.append(phi_eq)

        if mat_idx == 0:
            ax.plot(sizes_A, potentials_eq, label=f'This work', linestyle=linestyles[mat_idx], color='royalblue',lw=2)
        else:
            ax.plot(sizes_A, potentials_eq, linestyle=linestyles[mat_idx], color='royalblue',lw=2)

    legend1 = ax.legend(loc='upper right', fontsize=14, frameon=False)
    # Add legend for line styles
    from matplotlib.lines import Line2D
    line_handles = []
    line_labels = []
    for ls, label in zip(linestyles, ['graphite', 'silicate']):
        line_handles.append(Line2D([0], [0], color='k', linestyle=ls, lw=2))
        line_labels.append(label)
    legend2 = ax.legend(line_handles, line_labels, loc='upper left', fontsize=14, frameon=False)
    ax.add_artist(legend1)

    fig.subplots_adjust(top=0.99, bottom=0.14, left=0.1, right=0.99)
    out = _photoelectric_output_path(savefile or f'avg_potential_{radiation_model}.pdf')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    # --- additional plot: flux-weighted C_abs normalized by geometric cross-section ---
    from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections
    # rad may be returned in different formats (wavelength_nm, I_lambda) or (E_eV, I_E)
    rad_arr = np.asarray(rad)
    if rad_arr.ndim != 2:
        raise ValueError('Unexpected radiation field shape')
    conv = 1239.84193  # eV * nm
    if rad_arr.shape[1] == 2:
        # assume [wavelength_nm, intensity_per_nm]
        wav_nm = rad_arr[:, 0]
        I_lambda = rad_arr[:, 1]
        E_vals = conv / wav_nm
        # convert I_lambda (per nm) to I_E (per eV): I_E = I_lambda * |dλ/dE| = I_lambda * (λ^2 / conv)
        I_E = I_lambda * (wav_nm ** 2 / conv)
    else:
        col0 = rad_arr[:, 0]
        if np.nanmax(col0) <= 100.0:
            # first column is energy in eV
            E_vals = col0
            I_E = rad_arr[:, 2] if rad_arr.shape[1] > 2 else rad_arr[:, 1]
        else:
            # first column is wavelength in nm
            wav_nm = col0
            I_lambda = rad_arr[:, 1]
            E_vals = conv / wav_nm
            I_E = I_lambda * (wav_nm ** 2 / conv)
    mask_E = E_vals <= 13.6
    if not np.any(mask_E):
        raise ValueError('Radiation field has no points <= 13.6 eV')
    radE_sub = E_vals[mask_E]
    radI_sub = I_E[mask_E]

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'grain size $a$ [\AA]')
    ax2.set_ylabel(r'flux-weighted $\langle C_{\rm abs} \rangle / (\pi a^2)$')
    ax2.grid(True, which='both', ls=':', alpha=0.5)

    # Plot the data in Draine_Qabs_graphite.csv and Draine_Qabs_silicate.csv
    for mat in ['graphite', 'silicate']:
        if mat == 'graphite':
            data = np.loadtxt('Draine_Qabs_graphite.csv', delimiter=',', skiprows=1)
            linestyle=':'
        elif mat == 'silicate':
            data = np.loadtxt('Draine_Qabs_silicate.csv', delimiter=',', skiprows=1)
            linestyle='-.'
        sizes_draine = data[:, 0]  # in Angstroms
        Qabs_draine = data[:, 1]
        ax2.plot(sizes_draine, Qabs_draine, label=f'{mat} (Draine 2011)', linestyle=linestyle, color='k')

    for mat in materials:
        avgC = []
        for aA in sizes_A:
            a_micron = aA * 1e-4
            # interpolate cross sections (returns wavelengths in cm and C_abs in cm^2)
            a0, wav_cm, C_sca, C_abs, C_rp = interpolate_cross_sections(mat, a_micron, efficiency=True)
            # compute energy grid for C_abs: E(eV) = 1.2398e-4 / lambda(cm)
            E_vals = 1.2398e-4 / np.asarray(wav_cm)
            # interpolate C_abs onto radiation field energy grid (radE_sub)
            C_interp = np.interp(radE_sub[::-1], E_vals, np.asarray(C_abs))
            # flux-weighted average C_abs: integral I(E) C_abs(E) dE / integral I(E) dE
            num = np.trapezoid(radI_sub[::-1] * C_interp, radE_sub[::-1])
            den = np.trapezoid(radI_sub[::-1], radE_sub[::-1])
            if den <= 0:
                avgC.append(0.0)
            else:
                avgC.append(num / den)

        ax2.plot(sizes_A, avgC, label=f'{mat}')

    ax2.legend()
    fig2.tight_layout()
    out2 = _photoelectric_output_path(savefile.replace('.pdf', f'_fluxnorm_{radiation_model}.pdf') if savefile else f'avg_potential_{radiation_model}_fluxnorm.pdf')
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)

def plot_csa_IM19():
    use_calima_style()
    from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections

    # 1. Setup the figure
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_ylabel(r'$C_{\rm abs}$ [cm$^2$]', fontsize=16)
    ax.set_xlabel(r'$E$ [eV]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1,50)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()

    grain_sizes = np.array([3.5e-4,5e-4,10e-4,50e-4,100e-4,500e-4,1000e-4]) # in microns
    grain_types = ['graphite', 'silicate']
    line_colors = ['steelblue', 'sandybrown']
    line_styles = ['-','--',':','-.',(0, (1, 10)),(0, (1, 5)),(0, (3, 5, 1, 5, 1, 5))]
    for i in range(0, len(grain_types)):
        for j in range(0, len(grain_sizes)):
            a0,wav,_,C_abs,_ = interpolate_cross_sections(grain_types[i], grain_sizes[j])
            optical_E = 1.2398 / (wav*1e4)
            ax.plot(optical_E, C_abs, label=fr'{grain_types[i]}, $a={grain_sizes[j]*1e4:.0f}$ $\AA$',
                    color=line_colors[i],linestyle=line_styles[j], linewidth=2)
            
    # 2. Add legend and savefig
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    fig.savefig(_photoelectric_output_path('dust_absorption_cs_IM19.pdf'), format='pdf', dpi=300)


def compare_Mathis_WD01(savefile=None, E_min=0.1, E_max=13.6, nE=1000, norm_band=(5.17,13.6)):
    """Compare the Mathis (1983) field from the data file used by get_radiation_field
    with the Mathis-like implementation used in Weingartner & Draine (2001).

    This version compares absolute nu*u_nu (equivalently E * u_E) without
    normalizing the spectra. Both spectra are evaluated on a common energy grid
    and converted to an energy-density-per-eV basis (u_E in erg cm^-3 eV^-1).
    The plotted quantity is E * u_E (erg cm^-3).

    Returns the output filename.
    """
    use_calima_style()
    E = np.linspace(E_min, E_max, int(nE))
    # Evaluate WD01 Mathis function on the same E grid via get_radiation_field
    rad_WD = get_radiation_field('Mathis')
    # rad_WD is returned as [wavelength_nm, I_lambda] (by get_radiation_field for 'Mathis')
    rad_WD_arr = np.asarray(rad_WD)
    # convert rad_WD into u_E on the energy grid E
    conv = 1239.84193  # eV * nm
    if rad_WD_arr.ndim != 2 or rad_WD_arr.shape[1] != 2:
        raise ValueError('Unexpected shape for functional Mathis returned by get_radiation_field')
    wav_WD = rad_WD_arr[:, 0]
    I_lambda_WD = rad_WD_arr[:, 1]
    E_WD_file = conv / wav_WD
    I_E_WD_file = I_lambda_WD * (wav_WD ** 2 / conv)
    # interpolate onto E (ensure xp is ascending)
    idx_wd = np.argsort(E_WD_file)
    I_WD_onE = np.interp(E, E_WD_file[idx_wd], I_E_WD_file[idx_wd], left=0.0, right=0.0)
    c_cgs_local = 2.99792458e10
    u_E_WD = (4.0 * np.pi * I_WD_onE) / c_cgs_local

    # Load file-based Mathis explicitly
    rad = get_radiation_field('Mathis_file')
    rad_arr = np.asarray(rad)
    conv = 1239.84193  # eV * nm
    # convert file rad to I(E) per eV (erg / s / cm^2 / eV / sr or per-surface depending on file)
    if rad_arr.ndim != 2:
        raise ValueError('Unexpected radiation field shape from get_radiation_field')
    if rad_arr.shape[1] == 2:
        wav_nm = rad_arr[:, 0]
        I_lambda = rad_arr[:, 1]
        E_file = conv / wav_nm
        I_E_file = I_lambda * (wav_nm ** 2 / conv)
    else:
        col0 = rad_arr[:, 0]
        if np.nanmax(col0) <= 100.0:
            # already energy in eV in first column
            E_file = col0
            I_E_file = rad_arr[:, 2] if rad_arr.shape[1] > 2 else rad_arr[:, 1]
        else:
            wav_nm = col0
            I_lambda = rad_arr[:, 1]
            E_file = conv / wav_nm
            I_E_file = I_lambda * (wav_nm ** 2 / conv)

    # interpolate file-based I(E) onto E grid (ensure xp is ascending)
    idx_file = np.argsort(E_file)
    I_file_onE = np.interp(E, E_file[idx_file], I_E_file[idx_file], left=0.0, right=0.0)

    # Convert intensities I(E) [erg / s / cm^2 / eV] to energy density per eV u_E [erg / cm^3 / eV]
    c_cgs_local = 2.99792458e10
    # If file intensities are per steradian (many file fields are given per nm per sr), get_radiation_field
    # typically returns surface-integrated values; to be conservative multiply by 4π here to get energy density.
    u_E_file = (4.0 * np.pi * I_file_onE) / c_cgs_local

    # u_E_WD now contains the functional Mathis u_E interpolated onto E

    # Compute the quantity E * u_E (erg / cm^3) which is equivalent to nu * u_nu
    E_u_file = E * u_E_file
    E_u_WD = E * u_E_WD

    # Prepare plots: absolute curves (top) and ratio + absolute difference (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    ax1.loglog(E, E_u_file, label='Mathis file (E * u_E)')
    ax1.loglog(E, E_u_WD, label='Mathis WD01 (E * u_E)', linestyle='--')
    ax1.set_ylabel(r'$E \, u_E$ [erg cm$^{-3}$]')
    ax1.legend()
    ax1.grid(True, which='both', ls=':', alpha=0.5)

    ax1.legend()
    ax1.grid(True, which='both', ls=':', alpha=0.5)

    # Ratio and absolute difference
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(E_u_WD > 0, E_u_file / E_u_WD, np.nan)
    ax2.plot(E, ratio, color='tab:blue', label='File / WD01 (ratio)')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('File / WD01')
    ax2.axhline(1.0, color='k', linestyle=':')
    ax2.grid(True, ls=':', alpha=0.5)

    # absolute difference on second y-axis
    ax2b = ax2.twinx()
    abs_diff = E_u_file - E_u_WD
    ax2b.plot(E, abs_diff, color='tab:orange', alpha=0.7, label='File - WD01 (abs diff)')
    ax2b.set_ylabel(r'$(E\,u_E)_{\rm file} - (E\,u_E)_{\rm WD01}$ [erg cm$^{-3}$]', color='tab:orange')
    ax2b.tick_params(axis='y', labelcolor='tab:orange')

    # summary metrics
    L1 = np.trapezoid(np.abs(abs_diff), E)
    Linf = np.nanmax(np.abs(abs_diff))
    ax2.text(0.02, 0.95, f'L1={L1:.3e} erg cm-3 eV, Linf={Linf:.3e} erg cm-3', transform=ax2.transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Integrated energy density check (0 -> 13.6 eV) per WD01: 8.64e-13 erg/cm^3
    WD01_ref = 8.64e-13
    # Integrate u_E over E to get total energy density (erg/cm^3)
    total_u_file = np.trapezoid(u_E_file, E)
    total_u_WD = np.trapezoid(u_E_WD, E)

    txt = (f'Total energy density (0-13.6 eV):\n'
        f'  file: {total_u_file:.3e} erg/cm^3\n'
        f'  WD01 func: {total_u_WD:.3e} erg/cm^3\n'
        f'  WD01 ref: {WD01_ref:.3e} erg/cm^3')
    print(txt)

    # Annotate plot with totals and relative differences
    ax1.text(0.02, 0.03, f'File total: {total_u_file:.2e}\nWD01 total: {total_u_WD:.2e}\nWD01 ref: {WD01_ref:.2e}',
          transform=ax1.transAxes, fontsize=8, va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.7))
    # Warn if totals differ significantly from WD01 reference
    tol = 0.2  # 20% tolerance
    rel_err_WD = abs(total_u_WD - WD01_ref) / (WD01_ref + 1e-300)
    rel_err_file = abs(total_u_file - WD01_ref) / (WD01_ref + 1e-300)
    if rel_err_WD > tol:
        print(f'WARNING: Mathis WD01 function integrated value differs from WD01 ref by {rel_err_WD:.2%}')
    if rel_err_file > tol:
        print(f'WARNING: File-based Mathis integrated value differs from WD01 ref by {rel_err_file:.2%}')
    # Band-integrated check (6 -> 13.6 eV) per WD01: 6.07e-14 erg/cm^3
    WD01_band_ref = 6.07e-14
    band_mask = (E >= 6.0) & (E <= 13.6)
    if np.any(band_mask):
        total_u_file_band = np.trapezoid(u_E_file[band_mask], E[band_mask])
        total_u_WD_band = np.trapezoid(u_E_WD[band_mask], E[band_mask])
    else:
        total_u_file_band = 0.0
        total_u_WD_band = 0.0
    print(f'Band (6-13.6 eV) energy density: file={total_u_file_band:.3e}, WD01_func={total_u_WD_band:.3e}, WD01_ref={WD01_band_ref:.3e}')
    rel_err_band_WD = abs(total_u_WD_band - WD01_band_ref) / (WD01_band_ref + 1e-300)
    rel_err_band_file = abs(total_u_file_band - WD01_band_ref) / (WD01_band_ref + 1e-300)
    if rel_err_band_WD > tol:
        print(f'WARNING: Mathis WD01 function band (6-13.6 eV) differs from WD01 band ref by {rel_err_band_WD:.2%}')
    if rel_err_band_file > tol:
        print(f'WARNING: File-based Mathis band (6-13.6 eV) differs from WD01 band ref by {rel_err_band_file:.2%}')
    # annotate band totals on plot
    ax1.text(0.02, 0.12, f'Band(6-13.6eV): file {total_u_file_band:.2e}\nWD01 func {total_u_WD_band:.2e}\nref {WD01_band_ref:.2e}',
             transform=ax1.transAxes, fontsize=8, va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.7))

    out = _photoelectric_output_path(savefile or 'compare_Mathis_WD01_absolute.pdf')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def diagnose_Mathis_blackbody_models(E_min=0.01, E_max=13.6, nE=2000):
    """Diagnose the WD01 composite blackbody scaling.

    Compares two interpretations of the coefficients 1e-14, 1.65e-13, 4e-13:
      A) coefficients multiply B_nu (current implementation)
      B) coefficients multiply B_lambda; convert to B_nu before forming I_nu

    Returns a dict with integrated energy densities for both models and the
    file-based Mathis (using get_radiation_field) for comparison.
    """
    h_SI = 6.62607015e-34
    eV2J = 1.602176634e-19
    c_SI = 2.99792458e8

    E = np.linspace(E_min, E_max, int(nE))
    nu = E * eV2J / h_SI

    # Planck B_nu for each temperature (erg cm^-2 s^-1 Hz^-1 sr^-1)
    B_nu_7500 = Planck_function(7500, nu)
    B_nu_4000 = Planck_function(4000, nu)
    B_nu_3000 = Planck_function(3000, nu)

    # Model A: coefficients multiply B_nu
    I_nu_A = 1e-14 * B_nu_7500 + 1.65e-13 * B_nu_4000 + 4e-13 * B_nu_3000
    u_nu_A = (4.0 * np.pi * I_nu_A) / c_SI
    dnu_dE = eV2J / h_SI
    u_E_A = u_nu_A * dnu_dE
    total_u_A = np.trapezoid(u_E_A, E)

    # Model B: coefficients multiply B_lambda. compute B_lambda at lambda corresponding to E
    # lambda (m) = hc / E_J
    E_J = E * eV2J
    lam_m = h_SI * c_SI / E_J
    # B_lambda (SI): W m^-2 m^-1 sr^-1 = 2 h c^2 / lambda^5 / (exp(hc/(lambda kT)) -1)
    def B_lambda_SI(T, lam):
        exponent = np.exp(h_SI * c_SI / (lam * 1.380649e-23 * T))
        return (2.0 * h_SI * c_SI ** 2) / (lam ** 5) / (exponent - 1.0)

    B_l_7500 = B_lambda_SI(7500, lam_m)
    B_l_4000 = B_lambda_SI(4000, lam_m)
    B_l_3000 = B_lambda_SI(3000, lam_m)

    # Convert B_lambda (W m^-2 m^-1 sr^-1) -> B_lambda (erg cm^-2 s^-1 m^-1 sr^-1) multiply by 1e3
    B_l_7500_cgs = B_l_7500 * 1e3
    B_l_4000_cgs = B_l_4000 * 1e3
    B_l_3000_cgs = B_l_3000 * 1e3

    # Convert B_lambda to B_nu: B_nu = B_lambda * lambda^2 / c
    B_nu_from_l_7500 = B_l_7500_cgs * (lam_m ** 2) / c_SI
    B_nu_from_l_4000 = B_l_4000_cgs * (lam_m ** 2) / c_SI
    B_nu_from_l_3000 = B_l_3000_cgs * (lam_m ** 2) / c_SI

    # Now apply coefficients
    I_nu_B = 1e-14 * B_nu_from_l_7500 + 1.65e-13 * B_nu_from_l_4000 + 4e-13 * B_nu_from_l_3000
    u_nu_B = (4.0 * np.pi * I_nu_B) / c_SI
    u_E_B = u_nu_B * dnu_dE
    total_u_B = np.trapezoid(u_E_B, E)

    # Also compute file-based total using get_radiation_field (converted to u_E as compare function does)
    rad = get_radiation_field('Mathis')
    rad_arr = np.asarray(rad)
    conv = 1239.84193
    if rad_arr.shape[1] == 2:
        wav_nm = rad_arr[:, 0]
        I_lambda = rad_arr[:, 1]
        E_file = conv / wav_nm
        I_E_file = I_lambda * (wav_nm ** 2 / conv)
    else:
        col0 = rad_arr[:, 0]
        if np.nanmax(col0) <= 100.0:
            E_file = col0
            I_E_file = rad_arr[:, 2] if rad_arr.shape[1] > 2 else rad_arr[:, 1]
        else:
            wav_nm = col0
            I_lambda = rad_arr[:, 1]
            E_file = conv / wav_nm
            I_E_file = I_lambda * (wav_nm ** 2 / conv)
    I_file_onE = np.interp(E, E_file[::-1], I_E_file[::-1], left=0.0, right=0.0)
    c_cgs_local = 2.99792458e10
    u_E_file = (4.0 * np.pi * I_file_onE) / c_cgs_local
    total_u_file = np.trapezoid(u_E_file, E)

    results = {
        'total_u_A': total_u_A,
        'total_u_B': total_u_B,
        'total_u_file': total_u_file,
        'E_grid': E,
        'u_E_A': u_E_A,
        'u_E_B': u_E_B,
        'u_E_file': u_E_file,
    }
    print('Diagnose Mathis blackbody models:')
    print(f'  Model A (coeff * B_nu) total u = {total_u_A:.3e} erg/cm^3')
    print(f'  Model B (coeff * B_lambda->B_nu) total u = {total_u_B:.3e} erg/cm^3')
    print(f'  File-based total u = {total_u_file:.3e} erg/cm^3')
    # Band totals (6-13.6 eV)
    band_mask = (E >= 6.0) & (E <= 13.6)
    total_u_A_band = np.trapezoid(results['u_E_A'][band_mask], E[band_mask])
    total_u_B_band = np.trapezoid(results['u_E_B'][band_mask], E[band_mask])
    total_u_file_band = np.trapezoid(results['u_E_file'][band_mask], E[band_mask])
    print(f'  Model A band (6-13.6 eV) total u = {total_u_A_band:.3e} erg/cm^3')
    print(f'  Model B band (6-13.6 eV) total u = {total_u_B_band:.3e} erg/cm^3')
    print(f'  File-based band total u = {total_u_file_band:.3e} erg/cm^3')
    return results


def plot_heating_cooling_3d(grain_type, a_cm, radiation_model='Mathis', pair='G0_ne',
                           n_vals=32, rate='heating', savefile=None, show=True,
                           num_workers=None, T_groups=None, grain_label=None):
    """Compute a 3D surface for photoelectric heating or recombination cooling.

    The function returns (X, Y, Z, C) where C is the third variable used for colouring.

    Parameters
    ----------
    grain_type : str
        Grain material identifier passed to the equilibrium solver.
    a_cm : float
        Grain radius in cm.
    radiation_model : str
        Radiation model name used by equilibrium solver.
    pair : {'G0_ne','G0_sqrtT','sqrtT_ne'}
        Which two variables are used for the x/y axes. The third variable is encoded in the colour.
    n_x, n_y : int
        Grid resolution (keep modest for a smoke test).
    base_third : float
        Value to use for the third variable when it isn't being sampled. For example, when pair='G0_ne'
        base_third is sqrt(T).
    rate : {'heating','cooling'}
        Which scalar to compute for Z (heating or recombination cooling).
    savefile : str or None
        If provided, save a static snapshot to this path.
    show : bool
        If True, call plt.show() after plotting (interactive backends only).
    num_workers : int or None
        Number of parallel workers for ProcessPoolExecutor. Defaults to os.cpu_count().

    Returns
    -------
    X, Y, Z, C : np.ndarray
    """
    import concurrent.futures
    import time
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # lazy import of equilibrium solver to avoid startup cost when module imported
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

    # Enforce T_groups mode only (no gamma sweeps)
    if T_groups is None:
        raise ValueError('plot_heating_cooling_3d now requires T_groups (list of temperatures) and does not use gamma sweeps')

    if pair != 'G0_ne':
        raise ValueError('T_groups plotting currently only supported for pair=="G0_ne"')

    # ensure T_groups is an array
    T_groups = np.asarray(T_groups, dtype=float)

    # build G0/ne grids from n_vals
    nx = int(n_vals)
    ny = int(n_vals)
    x_vals = np.logspace(-3, 6, nx)   # G0
    y_vals = np.logspace(-4, 1, ny)   # ne
    

    # prepare figure with two 3D axes: left = heating surfaces per T, right = G0/ne vs T surface
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = _plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # color map for groups
    import matplotlib.cm as cm
    cmap = cm.get_cmap('plasma')

    results = {}
    # iterate over temperature groups and compute a surface for each
    # support an outer progress bar over temperatures
    outer_iter = T_groups
    try:
        from tqdm import tqdm
        outer_iter = tqdm(T_groups, desc='Temperatures', unit='T')
    except Exception:
        pass

    for ig, Tval in enumerate(outer_iter):
        tasks = []
        for j, y in enumerate(y_vals):
            for i, x in enumerate(x_vals):
                G0_used = float(x)
                ne_used = float(y)
                T_used = float(Tval)
                tasks.append((G0_used, ne_used, T_used, grain_type, a_cm, radiation_model))

        # evaluate in parallel and preserve order
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
            # inner progress bar for tasks
            try:
                from tqdm import tqdm as _tqdm
                results_iter = exe.map(_compute_rates_point, tasks)
                results_list = list(_tqdm(results_iter, total=len(tasks), desc=f'T={Tval:.3g}', unit='pts'))
            except Exception:
                try:
                    results_list = list(exe.map(_compute_rates_point, tasks))
                except Exception:
                    # fallback to serial
                    results_list = [ _compute_rates_point(t) for t in tasks ]

        # extract heating (first element) and reshape to grid (ny, nx)
        vals = np.array([r[0] for r in results_list], dtype=float)
        Z_mat = vals.reshape((ny, nx))

        # mesh for plotting
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

        # choose a color per temperature (convert to RGBA)
        color_rgba = cmap(float(ig) / max(1, len(T_groups)-1))

        # plot surface (log10 axes) on left axis
        surf = ax1.plot_surface(np.log10(X_grid), np.log10(Y_grid), np.log10(np.abs(Z_mat) + 1e-40),
                    color=color_rgba, alpha=0.6, linewidth=0, antialiased=False)

        # store this temperature's results so we can build the G0/ne vs T surface later
        results[float(Tval)] = {'G0': X_grid, 'ne': Y_grid, 'Z': Z_mat}

    ax1.set_xlabel('log10(G0)')
    ax1.set_ylabel('log10(ne)')
    ax1.set_zlabel(fr'log10({rate}) [erg s$^{{-1}}$]')

    # create legend with proxy artists for ax1
    from matplotlib.patches import Patch
    patches = [Patch(color=cmap(float(i)/max(1,len(T_groups)-1)), label=f'T={Tval:.1g} K') for i,Tval in enumerate(T_groups)]
    ax1.legend(handles=patches, loc='best')

    # Build the G0/ne vs T surface on ax2
    # x axis: log10(G0/ne) flattened over grid (length M = nx*ny)
    # y axis: log10(T) for each temperature (nT)
    nT = len(T_groups)
    M = nx * ny
    X_grid_all, Y_grid_all = np.meshgrid(x_vals, y_vals)
    x_flat = np.log10((X_grid_all / Y_grid_all).flatten())  # length M
    # build X2 (nT x M), Y2 (nT x M), Z2 (nT x M)
    X2 = np.tile(x_flat[None, :], (nT, 1))
    Y2 = np.tile(np.log10(T_groups)[:, None], (1, M))
    Z2_rows = []
    for Tval in T_groups:
        Z_mat = results[float(Tval)]['Z']
        Z2_rows.append(Z_mat.flatten())
    Z2 = np.vstack(Z2_rows)

    # Replace NaNs and zeros for log plotting
    eps = 1e-40
    Z2_plot = np.log10(np.abs(np.nan_to_num(Z2, nan=0.0)) + eps)

    # Plot the surface on ax2
    surf2 = ax2.plot_surface(X2, Y2, Z2_plot, cmap='viridis', alpha=0.7, linewidth=0, antialiased=False)
    ax2.set_xlabel('log10(G0/ne)')
    ax2.set_ylabel('log10(T)')
    ax2.set_zlabel(fr'log10({rate}) [erg s$^{{-1}}$]')
    fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1).set_label(fr'log10({rate})')

    if savefile:
        fig.savefig(_photoelectric_output_path(savefile), dpi=200)
    else:
        grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
        fig.savefig(_photoelectric_output_path(f'heating_cooling_3d_Tgroups_{grain_type}_{grain_tag}.pdf'), dpi=200)
    if show:
        _plt.show()
    _plt.close(fig)

    return results


def plot_heating_cooling_cartesian(grain_type, a_cm, radiation_model='Mathis',
                                  G0_vals=None, ne_vals=None, sqrtT_vals=None,
                                  rate='heating', save_prefix=None, show=True,
                                  num_workers=None, n_vals=10, grain_label=None):
    """Compute heating/cooling on the full Cartesian product of G0 x ne x sqrtT.

    Produces three scatter projection plots:
      - (G0, ne) color-coded by sqrtT
      - (G0, sqrtT) color-coded by ne
      - (sqrtT, ne) color-coded by G0

    Returns
    -------
    results : dict with keys 'G0','ne','sqrtT','heating','cooling'
    Arrays are flattened in the same order.
    """
    use_calima_style()
    import concurrent.futures
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # sensible defaults if not provided
    if G0_vals is None:
        G0_vals = np.logspace(-3, 6, n_vals)
    if ne_vals is None:
        ne_vals = np.logspace(-4, 1, n_vals)
    if sqrtT_vals is None:
        sqrtT_vals = np.logspace(1, 6, n_vals)

    # build tasks (Cartesian product)
    tasks = []
    metas = []
    for g0 in G0_vals:
        for ne in ne_vals:
            for sqrtT in sqrtT_vals:
                T = max(1e-8, float(sqrtT) ** 2)
                tasks.append((float(g0), float(ne), T, grain_type, a_cm, radiation_model))
                metas.append((g0, ne, sqrtT))

    N = len(tasks)
    heating = np.full(N, np.nan)
    cooling = np.full(N, np.nan)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(_compute_rates_point, t) for t in tasks]
        iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Cartesian grid') if tqdm is not None else concurrent.futures.as_completed(futures)
        for idx, fut in enumerate(iterator):
            peh, rec = fut.result()
            heating[idx] = peh
            cooling[idx] = rec

    # assemble arrays in same order as metas (g0 fastest, then ne, then sqrtT)
    G0_arr = np.array([m[0] for m in metas])
    ne_arr = np.array([m[1] for m in metas])
    sqrtT_arr = np.array([m[2] for m in metas])

    results = {'G0': G0_arr, 'ne': ne_arr, 'sqrtT': sqrtT_arr, 'heating': heating, 'cooling': cooling}

    # plotting projections
    import matplotlib as mpl
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sc0 = axes[0].scatter(G0_arr, ne_arr, c=sqrtT_arr, norm=mpl.colors.LogNorm(), cmap='viridis', s=8)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('G0'); axes[0].set_ylabel('ne'); axes[0].set_title(f'G0 vs ne ({rate})')
    cbar0 = fig.colorbar(sc0, ax=axes[0]); cbar0.set_label('sqrtT')

    sc1 = axes[1].scatter(G0_arr, sqrtT_arr, c=ne_arr, norm=mpl.colors.LogNorm(), cmap='plasma', s=8)
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('G0'); axes[1].set_ylabel('sqrtT'); axes[1].set_title(f'G0 vs sqrtT ({rate})')
    cbar1 = fig.colorbar(sc1, ax=axes[1]); cbar1.set_label('ne')

    sc2 = axes[2].scatter(sqrtT_arr, ne_arr, c=G0_arr, norm=mpl.colors.LogNorm(), cmap='inferno', s=8)
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlabel('sqrtT'); axes[2].set_ylabel('ne'); axes[2].set_title(f'sqrtT vs ne ({rate})')
    cbar2 = fig.colorbar(sc2, ax=axes[2]); cbar2.set_label('G0')

    if save_prefix:
        grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
        outname = f'{save_prefix}_cartesian_{grain_type}_{grain_tag}.pdf'
        outname = _photoelectric_output_path(outname)
        fig.savefig(outname, dpi=200)
        print(f'Saved cartesian projection plots to {outname}')
    if show:
        plt.show()
    plt.close(fig)

    return results


def plot_heating_cooling_cartesian_3d(grain_type, a_cm, radiation_model='Mathis',
                                     G0_vals=None, ne_vals=None, T_vals=None,
                                     rate='heating', save_prefix=None, show=True,
                                     num_workers=None, n_vals=10, point_size=12, cmap='viridis',
                                     grain_label=None):
    """Compute heating/cooling on the full Cartesian product and plot a 3D scatter.

    x/y are two chosen axes encoded by which arrays the user provides (G0_vals, ne_vals, sqrtT_vals).
    The function places the three axes as X=G0, Y=ne, Z=heating (or cooling) and colours points by sqrtT.

    Returns the same results dict as `plot_heating_cooling_cartesian`.
    """
    import concurrent.futures
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # sensible defaults
    if G0_vals is None:
        G0_vals = np.logspace(-3, 6, n_vals)
    if ne_vals is None:
        ne_vals = np.logspace(-4, 1, n_vals)
    if T_vals is None:
        T_vals = np.logspace(1, 6, n_vals)

    # build tasks
    tasks = []
    metas = []
    for g0 in G0_vals:
        for ne in ne_vals:
            for T in T_vals:
                tasks.append((float(g0), float(ne), T, grain_type, a_cm, radiation_model))
                metas.append((g0, ne, T))

    N = len(tasks)
    heating = np.full(N, np.nan)
    cooling = np.full(N, np.nan)

    # execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(_compute_rates_point, t) for t in tasks]
        iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Cartesian 3D') if tqdm is not None else concurrent.futures.as_completed(futures)
        for idx, fut in enumerate(iterator):
            peh, rec = fut.result()
            heating[idx] = peh
            cooling[idx] = rec

    # assemble arrays
    G0_arr = np.array([m[0] for m in metas])
    ne_arr = np.array([m[1] for m in metas])
    T_arr = np.array([m[2] for m in metas])

    results = {'G0': G0_arr, 'ne': ne_arr, 'T': T_arr, 'heating': heating, 'cooling': cooling}

    # Prepare 3D scatter: x=G0, y=ne, z=rate, color=third var (T)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as _plt
    import matplotlib.tri as mtri
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    X = G0_arr
    Y = ne_arr
    Z = heating if rate == 'heating' else cooling
    C = T_arr

    fig = _plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # convert to log space for plotting
    x_log = np.log10(X)
    y_log = np.log10(Y)
    z_log = np.log10(np.abs(Z) + 1e-40)

    # color normalization for sqrt(T): use LogNorm when values are strictly positive
    # --- normalization ---
    finite_mask = np.isfinite(C)
    Cpos_mask = finite_mask & (C > 0)

    if np.any(Cpos_mask):
        norm = mcolors.LogNorm(vmin=np.nanmin(C[Cpos_mask]), vmax=np.nanmax(C[Cpos_mask]))
    else:
        norm = mcolors.Normalize(
            vmin=np.nanmin(C[finite_mask]) if np.any(finite_mask) else 0.0,
            vmax=np.nanmax(C[finite_mask]) if np.any(finite_mask) else 1.0
        )

    # --- triangulation ---
    tri = mtri.Triangulation(x_log, y_log)

    # --- per-triangle averaged values ---
    try:
        tri_vals = np.nanmean(C[tri.triangles], axis=1)
    except Exception:
        tri_vals = C

    # --- colormap ---
    cmap_obj = cm.get_cmap(cmap)
    facecolors = cmap_obj(norm(tri_vals))

    # --- manually build the surface ---
    verts = np.array([
        list(zip(x_log[tri.triangles[i]],
                y_log[tri.triangles[i]],
                z_log[tri.triangles[i]]))
        for i in range(len(tri.triangles))
    ])

    surf = Poly3DCollection(verts, facecolors=facecolors, linewidths=0.2, antialiased=True)
    ax.add_collection3d(surf)
    ax.auto_scale_xyz(x_log, y_log, z_log)

    # --- colorbar ---
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(tri_vals)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label('log10(T)')

    ax.set_xlabel('log10(G0)')
    ax.set_ylabel('log10(ne)')
    ax.set_zlabel(fr'log10({rate}) [erg s$^{{-1}}$]')

    if save_prefix:
        grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
        outname = f'{save_prefix}_cartesian_3d_{grain_type}_{grain_tag}.pdf'
        outname = _photoelectric_output_path(outname)
        fig.savefig(outname, dpi=200)
        print(f'Saved 3D cartesian plot to {outname}')
    if show:
        _plt.show()
    _plt.close(fig)

    return results


def plot_gamma_combo_projections(grain_type, a_cm, radiation_model='Mathis',
                                 G0_vals=None, ne_vals=None, T_vals=None,
                                 rate='heating', save_prefix=None, show=True,
                                 num_workers=None, n_vals=12, nbins=50, grain_label=None):
    """Evaluate heating/cooling on Cartesian grid and plot deviations from gamma-only trend.

    gamma is defined as: gamma = G0 * sqrt(T) / ne

    The function computes heating (or cooling) for the full Cartesian product, then computes
    a median trend heating_med(gamma) by binning in gamma. The deviation for each point is
    defined as log10(heating) - log10(median_heating_at_same_gamma).

    Produces three scatter plots:
      - gamma vs G0 (color = deviation)
      - gamma vs ne (color = deviation)
      - gamma vs T  (color = deviation)

    Returns the results dict used for plotting.
    """
    use_calima_style()
    import concurrent.futures
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # sensible defaults
    if G0_vals is None:
        G0_vals = np.logspace(-3, 6, n_vals)
    if ne_vals is None:
        ne_vals = np.logspace(-4, 1, n_vals)
    if T_vals is None:
        T_vals = np.logspace(1, 6, n_vals)

    # build tasks and metas
    tasks = []
    metas = []
    for g0 in G0_vals:
        for ne in ne_vals:
            for T in T_vals:
                tasks.append((float(g0), float(ne), float(T), grain_type, a_cm, radiation_model))
                metas.append((g0, ne, T))

    N = len(tasks)
    heating = np.full(N, np.nan)
    cooling = np.full(N, np.nan)

    # execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(_compute_rates_point, t) for t in tasks]
        iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Gamma combo') if tqdm is not None else concurrent.futures.as_completed(futures)
        for idx, fut in enumerate(iterator):
            peh, rec = fut.result()
            heating[idx] = peh
            cooling[idx] = rec

    # assemble arrays
    G0_arr = np.array([m[0] for m in metas])
    ne_arr = np.array([m[1] for m in metas])
    T_arr = np.array([m[2] for m in metas])
    rate_arr = heating if rate == 'heating' else cooling

    # compute gamma combination
    gamma = (G0_arr * np.sqrt(T_arr)) / ne_arr

    # compute median trend in log-space by binning gamma
    mask = np.isfinite(rate_arr) & (rate_arr > 0) & np.isfinite(gamma) & (gamma > 0)
    log_rate = np.log10(rate_arr[mask])
    log_gamma = np.log10(gamma[mask])

    if len(log_gamma) == 0:
        raise RuntimeError('No finite positive points found for computing gamma trend')

    bins = np.logspace(np.nanmin(log_gamma), np.nanmax(log_gamma), nbins)
    bin_idx = np.digitize(np.power(10.0, log_gamma), bins)
    medians = {}
    for b in range(1, len(bins)+1):
        sel = bin_idx == b
        if np.any(sel):
            medians[b] = np.median(log_rate[sel])

    # map each point to the median at its gamma bin
    mapped_median = np.full_like(log_rate, np.nan)
    for b, med in medians.items():
        mapped_median[bin_idx == b] = med

    deviation = np.full_like(rate_arr, np.nan)
    # fill deviation only for masked points
    deviation[mask] = log_rate - mapped_median

    # plotting: three panels
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # scatter gamma vs G0
    sc0 = axes[0].scatter(G0_arr, gamma, c=deviation, cmap='RdBu_r', vmin=-np.nanmax(np.abs(deviation)), vmax=np.nanmax(np.abs(deviation)), s=8)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('G0'); axes[0].set_ylabel('gamma'); axes[0].set_title('gamma vs G0 (deviation)')
    fig.colorbar(sc0, ax=axes[0]).set_label('log10(dev)')

    # scatter gamma vs ne
    sc1 = axes[1].scatter(ne_arr, gamma, c=deviation, cmap='RdBu_r', vmin=-np.nanmax(np.abs(deviation)), vmax=np.nanmax(np.abs(deviation)), s=8)
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('ne'); axes[1].set_ylabel('gamma'); axes[1].set_title('gamma vs ne (deviation)')
    fig.colorbar(sc1, ax=axes[1]).set_label('log10(dev)')

    # scatter gamma vs T
    sc2 = axes[2].scatter(T_arr, gamma, c=deviation, cmap='RdBu_r', vmin=-np.nanmax(np.abs(deviation)), vmax=np.nanmax(np.abs(deviation)), s=8)
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlabel('T'); axes[2].set_ylabel('gamma'); axes[2].set_title('gamma vs T (deviation)')
    fig.colorbar(sc2, ax=axes[2]).set_label('log10(dev)')

    if save_prefix:
        grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
        outname = f'{save_prefix}_gamma_combo_{grain_type}_{grain_tag}.pdf'
        outname = _photoelectric_output_path(outname)
        fig.savefig(outname, dpi=200)
        print(f'Saved gamma combo plots to {outname}')
    if show:
        plt.show()
    plt.close(fig)

    results = {'G0': G0_arr, 'ne': ne_arr, 'T': T_arr, 'gamma': gamma, 'rate': rate_arr, 'deviation': deviation}
    return results


def plot_gamma_surfaces(grain_type, a_cm, radiation_model='Mathis',
                        G0_vals=None, ne_vals=None, T_vals=None,
                        rate='heating', save_prefix=None, show=True,
                        num_workers=None, n_vals=12, cmap='viridis', grain_label=None):
    """Compute gamma = G0*sqrt(T)/ne on a Cartesian grid and plot 3D surfaces.

    Produces three 3D panels where the x-axis is the variable (G0, ne or T), the y-axis
    is gamma, and the z-axis is the log10(rate) (heating or cooling).
    """
    use_calima_style()
    import concurrent.futures
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # sensible defaults
    if G0_vals is None:
        G0_vals = np.logspace(-3, 6, n_vals)
    if ne_vals is None:
        ne_vals = np.logspace(-4, 1, n_vals)
    if T_vals is None:
        T_vals = np.logspace(1, 6, n_vals)

    # build tasks
    tasks = []
    metas = []
    for g0 in G0_vals:
        for ne in ne_vals:
            for T in T_vals:
                tasks.append((float(g0), float(ne), float(T), grain_type, a_cm, radiation_model))
                metas.append((g0, ne, T))

    N = len(tasks)
    heating = np.full(N, np.nan)
    cooling = np.full(N, np.nan)

    # run tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(_compute_rates_point, t) for t in tasks]
        iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Gamma surfaces') if tqdm is not None else concurrent.futures.as_completed(futures)
        for idx, fut in enumerate(iterator):
            peh, rec = fut.result()
            heating[idx] = peh
            cooling[idx] = rec

    # assemble arrays
    G0_arr = np.array([m[0] for m in metas])
    ne_arr = np.array([m[1] for m in metas])
    T_arr = np.array([m[2] for m in metas])
    rate_arr = heating if rate == 'heating' else cooling

    # compute gamma
    gamma = (G0_arr * np.sqrt(T_arr)) / ne_arr

    # Prepare plotting
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.tri as mtri
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]
    titles = ['G0', 'ne', 'T']
    var_arrays = [G0_arr, ne_arr, T_arr]

    # compute z values (log10) and masks
    eps = 1e-40
    z_vals = np.log10(np.abs(rate_arr) + eps)

    for ax, title, var in zip(axes, titles, var_arrays):
        # x = var, y = gamma, z = z_vals
        # convert to log space for x and y
        x = np.log10(var)
        y = np.log10(gamma)
        z = z_vals

        # triangulate in x-y
        try:
            tri = mtri.Triangulation(x, y)
        except Exception:
            # fallback to scatter if triangulation fails
            sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=6)
            fig.colorbar(sc, ax=ax, shrink=0.6).set_label(fr'log10({rate})')
            ax.set_xlabel(f'log10({title})')
            ax.set_ylabel('log10(gamma)')
            ax.set_zlabel(fr'log10({rate})')
            continue

        # per-triangle averaged color (use mean z per triangle)
        try:
            tri_vals = np.nanmean(z[tri.triangles], axis=1)
        except Exception:
            tri_vals = z

        cmap_obj = cm.get_cmap(cmap)
        # use symetric color range around median for visibility
        vmin = np.nanpercentile(tri_vals, 2)
        vmax = np.nanpercentile(tri_vals, 98)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        facecolors = cmap_obj(norm(tri_vals))

        verts = np.array([
            list(zip(x[tri.triangles[i]], y[tri.triangles[i]], z[tri.triangles[i]]))
            for i in range(len(tri.triangles))
        ])

        surf = Poly3DCollection(verts, facecolors=facecolors, linewidths=0.0, antialiased=True)
        ax.add_collection3d(surf)
        ax.auto_scale_xyz(x, y, z)

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(tri_vals)
        fig.colorbar(mappable, ax=ax, shrink=0.6).set_label(fr'log10({rate})')

        ax.set_xlabel(f'log10({title})')
        ax.set_ylabel('log10(gamma)')
        ax.set_zlabel(fr'log10({rate})')

    if save_prefix:
        grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
        outname = f'{save_prefix}_gamma_surfaces_{grain_type}_{grain_tag}.pdf'
        outname = _photoelectric_output_path(outname)
        fig.savefig(outname, dpi=200)
        print(f'Saved gamma surfaces to {outname}')
    if show:
        plt.show()
    plt.close(fig)

    return {'G0': G0_arr, 'ne': ne_arr, 'T': T_arr, 'gamma': gamma, 'rate': rate_arr}


def plot_rate_vs_gamma_for_T(grain_type, a_cm, radiation_model='Mathis',
                             mode='fix_G0', fixed_value=1.0,
                             gamma_vals=None, T_groups=None,
                             rate='heating', save_prefix=None, show=True,
                             num_workers=None, n_gamma=50, grain_label=None):
    """Compute heating (or cooling) vs gamma for different T while fixing G0 or ne.

    Parameters
    ----------
    mode : {'fix_G0','fix_ne'}
        If 'fix_G0', `fixed_value` is the G0 to hold constant and ne is derived as ne = G0 * sqrt(T) / gamma.
        If 'fix_ne', `fixed_value` is the ne to hold constant and G0 is derived as G0 = gamma * ne / sqrt(T).
    gamma_vals : array-like or None
        Values of gamma to sample. If None, use logspace from 1e-6 to 1e6 with n_gamma points.
    T_groups : list-like
        Temperatures to compute curves for.

    Returns
    -------
    results: dict with keys for each T -> dict(gamma, rate)
    """
    use_calima_style()
    import concurrent.futures
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if gamma_vals is None:
        gamma_vals = np.logspace(-2, 6, n_gamma)
    else:
        gamma_vals = np.array(gamma_vals, dtype=float)

    if T_groups is None:
        T_groups = [10.0, 100.0, 1000.0]
    T_groups = np.asarray(T_groups, dtype=float)

    tasks = []
    metas = []  # (T, gamma)

    for T in T_groups:
        sqrtT = np.sqrt(T)
        for gamma in gamma_vals:
            if mode == 'fix_G0':
                G0 = float(fixed_value)
                # ne = G0 * sqrt(T) / gamma
                ne = max(1e-20, (G0 * sqrtT) / float(gamma))
            elif mode == 'fix_ne':
                ne = float(fixed_value)
                # G0 = gamma * ne / sqrt(T)
                G0 = max(1e-20, (float(gamma) * ne) / sqrtT)
            else:
                raise ValueError('mode must be "fix_G0" or "fix_ne"')

            tasks.append((G0, ne, float(T), grain_type, a_cm, radiation_model))
            metas.append((T, gamma))

    N = len(tasks)
    peh_vals = np.full(N, np.nan)
    rec_vals = np.full(N, np.nan)
    Zmean_vals = np.full(N, np.nan)
    Zsigma_vals = np.full(N, np.nan)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(_compute_rates_point, t) for t in tasks]
        iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Rate vs gamma') if tqdm is not None else concurrent.futures.as_completed(futures)
        for idx, fut in enumerate(iterator):
            peh, rec, Zmean, Zsigma = fut.result()
            peh_vals[idx] = peh
            rec_vals[idx] = rec
            Zmean_vals[idx] = Zmean
            Zsigma_vals[idx] = Zsigma

    # assemble results per T
    results = {}
    idx = 0
    for T in T_groups:
        peh = peh_vals[idx: idx + len(gamma_vals)]
        rec = rec_vals[idx: idx + len(gamma_vals)]
        Zmean = Zmean_vals[idx: idx + len(gamma_vals)]
        Zsigma = Zsigma_vals[idx: idx + len(gamma_vals)]
        results[float(T)] = {
            'gamma': gamma_vals.copy(),
            'peh': peh.copy(),
            'rec': rec.copy(),
            'Zmean': Zmean.copy(),
            'Zsigma': Zsigma.copy()
        }
        idx += len(gamma_vals)

    # plotting
    import matplotlib.pyplot as plt
    fig, ax_heating = plt.subplots(1, 1, figsize=(8, 6))
    ax_cooling = ax_heating.twinx()

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(T_groups) - 1)) for i in range(len(T_groups))]

    # running median helper
    def running_median(y, width):
        from collections import deque
        import bisect
        n = len(y)
        if n <= 3 or width <= 1:
            return y.copy()
        w = max(1, int(width))
        half = w // 2
        out = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = np.sort(y[lo:hi])
            out[i] = np.median(window)
        return out

    for iT, T in enumerate(T_groups):
        res = results[float(T)]
        gamma = res['gamma']
        peh = res['peh']
        rec = res['rec']
        # apply running median smoothing; window scales with gamma length (use ~5% window)
        win = max(3, int(np.ceil(0.05 * max(3, len(gamma)))))
        peh_smooth = running_median(peh, win)
        rec_smooth = running_median(rec, win)
        # safe log plots
        peh_plot = np.log10(np.abs(peh_smooth) + 1e-40)
        rec_plot = np.log10(np.abs(rec_smooth) + 1e-40)
        color = colors[iT]
        ax_heating.plot(gamma, peh_plot, label=f'Heating T={T:.1g} K', color=color, ls='-')
        ax_cooling.plot(gamma, rec_plot, label=f'Cooling T={T:.1g} K', color=color, ls='--', alpha=0.9)

    ax_heating.set_xscale('log')
    ax_heating.set_xlabel('gamma = G0 * sqrt(T) / ne')
    ax_heating.set_ylabel(r'log10(heating) [erg s$^{-1}$]', color='tab:blue')
    ax_cooling.set_ylabel(r'log10(recombination cooling) [erg s$^{-1}$]', color='tab:orange')

    # Build combined legend
    handles_h, labels_h = ax_heating.get_legend_handles_labels()
    handles_c, labels_c = ax_cooling.get_legend_handles_labels()
    # show heating and cooling entries together
    ax_heating.legend(handles_h + handles_c, labels_h + labels_c, loc='best', fontsize='small')
    ax_heating.grid(True, which='both', ls='--', lw=0.5)

    if save_prefix:
        grain_tag = _grain_output_label(a_cm=a_cm, grain_label=grain_label)
        outname = f'{save_prefix}_rate_vs_gamma_{mode}_{grain_type}_{grain_tag}.pdf'
        outname = _photoelectric_output_path(outname)
        fig.savefig(outname, dpi=200)
        print(f'Saved rate vs gamma figure to {outname}')
    if show:
        plt.show()
    plt.close(fig)

    return results