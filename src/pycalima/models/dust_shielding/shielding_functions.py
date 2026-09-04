"""
Self-shielding factors for H2, CO, and dust.

Ports the shielding physics from RAMSES rtz/molecules_module.f90 to Python,
and adds a new per-bin dust shielding path for the CALIMA dust evolution model.

Physical context
----------------
H2 Lyman-Werner self-shielding reduces the photodissociation rate when the H2
column becomes large enough to saturate the LW lines.  Dust in the ISM
additionally attenuates the FUV field continuum.  With the CALIMA dust evolution
model the grain size distribution and composition vary per cell, changing the
effective LW opacity per unit dust mass (κ_LW).  This module:

  1. Reproduces the RAMSES ``comp_SH2`` and ``comp_Sd`` functions for
     comparison/testing (``comp_SH2``, ``comp_Sd_old``).
  2. Provides a new implementation that uses per-bin κ_LW from pyCALIMA's
     optical property tables (``comp_Sd_new``).
  3. Provides ``comp_SCO`` — a port of the CO self-shielding table look-up from
     RAMSES molecules_module.f90 (Lee 1996 / Visser et al. 2009 tables).
  4. Provides ``comp_G0_selfshield`` — dust self-shielding of the radiation
     field, using the same τ_dust philosophy applied to G0 fed to CALIMA.
  5. Provides ``compute_kappa_LW_draine`` and ``compute_kappa_LW_precomp`` to
     obtain κ_LW [cm²/g_dust] for arbitrary grain sizes / the default CALIMA
     bins respectively.
"""

import os
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from pycalima import _paths
from pycalima.models.grain_size_config import get_model_data_dir

# ─────────────────────────────── constants ────────────────────────────────────
m_H   = 1.6726e-24   # proton mass [g]
h_cgs = 6.6261e-27   # Planck constant [erg s]
c_cgs = 2.9979e10    # speed of light [cm/s]
eV_to_erg = 1.6022e-12

# Old RAMSES "bare grain-silicate" dust cross-section per H atom at LW band
SDEFF_MW = 2.34e-21   # cm²/H-atom  (Gnedin & Kravtsov 2009, eq. in comp_Sd)

# Canonical MW dust-to-gas ratio (mass fraction) used in RAMSES
DGR_MW   = 1.0 / 150.0   # ≈ 0.67% — RAMSES uses dust_to_gas_scale_RR14 which
                           # normalises to this value at Z_sun.  The exact value
                           # does not affect the shielding physics but sets the
                           # conversion between σ_eff and κ_LW below.

# Effective LW opacity implied by the old formula:
#   τ_old = σ_eff × Z × N_H  =  κ_LW_eff × ρ_dust × dx
#   ρ_dust = Z × DGR_MW × n_H × m_H
#   → κ_LW_eff = SDEFF_MW / (DGR_MW × m_H)
KAPPA_LW_MW_EFF = SDEFF_MW / (DGR_MW * m_H)  # cm²/g_dust

# LW / FUV band limits used throughout
E_LW_MIN = 6.0    # eV  (Habing G0 lower edge)
E_LW_MAX = 13.6   # eV  (H ionisation threshold)

# ──────────────────────────────── paths ───────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_OPTICS_DIR  = _paths.get_optical_props_path("draine_lee_1984")
_MODEL_DIR   = get_model_data_dir() / "optical_properties"

# ═══════════════════════════════════════════════════════════════════════════════
#   H2 self-shielding
# ═══════════════════════════════════════════════════════════════════════════════

def comp_SH2(N_H2):
    """H2 Lyman-Werner self-shielding factor.

    Port of RAMSES ``rtz/molecules_module.f90 :: comp_SH2``.
    Analytic fit from Draine & Bertoldi (1996), see also
    Gnedin & Kravtsov (2009) Appendix A.

    Parameters
    ----------
    N_H2 : float or array-like
        H2 column density [cm⁻²].

    Returns
    -------
    ndarray
        Shielding factor ∈ (0, 1].  At N_H2 = 0, returns 1.
    """
    N_H2 = np.asarray(N_H2, dtype=float)
    x    = N_H2 / 5.0e14          # dimensionless column
    w    = 0.2                     # H2 line damping weight
    Sa   = (1.0 - w) / (1.0 + x)**2
    Sb   = w / np.sqrt(1.0 + x)
    Sc   = np.exp(-8.5e-4 * np.sqrt(1.0 + x))
    return Sa + Sb * Sc


# ═══════════════════════════════════════════════════════════════════════════════
#   Dust LW shielding — old formula (current RAMSES)
# ═══════════════════════════════════════════════════════════════════════════════

def comp_Sd_old(N_HI, N_H2, Z=1.0):
    """Dust attenuation of the LW field — current RAMSES implementation.

    Port of RAMSES ``rtz/molecules_module.f90 :: comp_Sd``.
    Gnedin & Kravtsov (2009), http://iopscience.iop.org/0004-637X/697/1/55

    Uses a fixed dust cross-section per H atom
    ``SDEFF_MW = 2.34×10⁻²¹ cm²`` (bare graphite-silicate MW mix)
    scaled by ``Z``, the dust-to-gas ratio relative to MW.

    Parameters
    ----------
    N_HI : float or array-like
        Neutral hydrogen column density [cm⁻²].
    N_H2 : float or array-like
        H2 column density [cm⁻²] (H2 molecules, not H atoms).
    Z : float or array-like
        Dust-to-gas ratio relative to MW (1 = MW, <1 = dust-poor).

    Returns
    -------
    ndarray
        Dust LW shielding factor ∈ (0, 1].
    """
    N_HI = np.asarray(N_HI, dtype=float)
    N_H2 = np.asarray(N_H2, dtype=float)
    Z    = np.asarray(Z,    dtype=float)
    # N_HI + 2*N_H2 = total hydrogen column weighted by shielding geometry
    tau  = SDEFF_MW * Z * (N_HI + 2.0 * N_H2)
    return np.exp(-tau)


# ═══════════════════════════════════════════════════════════════════════════════
#   Dust LW shielding — new formula (per-bin CALIMA)
# ═══════════════════════════════════════════════════════════════════════════════

def comp_Sd_new(tau_dust_LW):
    """Dust LW shielding using a pre-computed per-cell optical depth.

    New formulation for RAMSES+CALIMA: replaces the fixed ``SDEFF_MW`` with
    τ_dust computed from per-bin grain optical properties:

        τ_dust_LW = Σ_i  κ_LW,i × ρ_dust,i × dx

    where κ_LW,i [cm²/g] is the LW-band mass opacity of bin i.

    Parameters
    ----------
    tau_dust_LW : float or array-like
        Dust optical depth in the LW band (dimensionless, ≥ 0).

    Returns
    -------
    ndarray
        Shielding factor exp(-τ_dust_LW) ∈ (0, 1].
    """
    return np.exp(-np.asarray(tau_dust_LW, dtype=float))


def compute_tau_dust_LW(rho_dust_bins, kappa_LW_bins, dx):
    """Compute the LW-band dust optical depth for a single cell.

    Parameters
    ----------
    rho_dust_bins : array-like, shape (nbins,)
        Dust mass density per bin [g/cm³].
    kappa_LW_bins : array-like, shape (nbins,)
        LW-band mass opacity per bin [cm²/g].
    dx : float
        Path length through the cell [cm].

    Returns
    -------
    float
        τ_dust_LW = Σ_i κ_LW,i × ρ_dust,i × dx.
    """
    rho = np.asarray(rho_dust_bins, dtype=float)
    kap = np.asarray(kappa_LW_bins, dtype=float)
    return float(np.sum(kap * rho) * dx)


# ═══════════════════════════════════════════════════════════════════════════════
#   CO self-shielding
# ═══════════════════════════════════════════════════════════════════════════════

# ── CO self-shielding table (N_CO vs θ_CO) ─────────────────────────────────
# Ported directly from RAMSES molecules_module.f90 :: initialize_SCO_table
# Lee et al. (1996) / Visser et al. (2009) lookup table
# sco_table[:,0] = CO column density [cm⁻²]
# sco_table[:,1] = CO self-shielding factor
_SCO_TABLE = np.array([
    [1.000e+00, 1.000e+00],
    [1.000e+12, 9.990e-01], [1.650e+12, 9.981e-01], [2.995e+12, 9.961e-01],
    [5.979e+12, 9.912e-01], [1.313e+13, 9.815e-01], [3.172e+13, 9.601e-01],
    [8.429e+13, 9.113e-01], [2.464e+14, 8.094e-01], [7.923e+14, 6.284e-01],
    [1.670e+15, 4.808e-01], [2.595e+15, 3.889e-01], [4.435e+15, 2.827e-01],
    [6.008e+15, 2.293e-01], [8.952e+15, 1.695e-01], [1.334e+16, 1.224e-01],
    [1.661e+16, 1.017e-01], [2.274e+16, 7.764e-02], [3.115e+16, 5.931e-02],
    [4.266e+16, 4.546e-02], [5.843e+16, 3.506e-02], [8.002e+16, 2.728e-02],
    [1.096e+17, 2.143e-02], [1.501e+17, 1.700e-02], [2.055e+17, 1.360e-02],
    [2.815e+17, 1.094e-02], [4.241e+17, 8.273e-03], [6.389e+17, 6.283e-03],
    [9.625e+17, 4.773e-03], [1.450e+18, 3.611e-03], [2.184e+18, 2.704e-03],
    [3.291e+18, 1.986e-03], [4.124e+18, 1.657e-03], [5.685e+18, 1.258e-03],
    [7.838e+18, 9.332e-04], [1.080e+19, 6.745e-04], [1.285e+19, 5.596e-04],
    [1.681e+19, 4.123e-04], [2.199e+19, 2.982e-04], [2.538e+19, 2.490e-04],
    [3.222e+19, 1.827e-04], [4.091e+19, 1.324e-04], [5.193e+19, 9.473e-05],
    [5.893e+19, 7.891e-05], [7.356e+19, 5.668e-05], [8.269e+19, 4.732e-05],
    [9.246e+19, 3.967e-05], [1.031e+20, 3.327e-05], [1.148e+20, 2.788e-05],
    [1.277e+20, 2.331e-05], [1.419e+20, 1.944e-05], [1.578e+20, 1.619e-05],
])

# ── H2 continuum shielding of CO photodissociation ──────────────────────────
# sh2_table[:,0] = H2 column density [cm⁻²]
# sh2_table[:,1] = H2 continuum shielding factor for CO photodissociation
_SH2_CO_TABLE = np.array([
    [1.000e+00, 1.000e+00],
    [2.666e+13, 9.999e-01], [3.801e+14, 9.893e-01], [6.634e+15, 9.678e-01],
    [8.829e+16, 9.465e-01], [9.268e+17, 9.137e-01], [1.007e+18, 9.121e-01],
    [2.021e+18, 8.966e-01], [3.036e+18, 8.862e-01], [4.051e+18, 8.781e-01],
    [5.066e+18, 8.716e-01], [6.082e+18, 8.660e-01], [7.097e+18, 8.612e-01],
    [8.112e+18, 8.569e-01], [9.341e+18, 8.524e-01], [1.014e+19, 8.497e-01],
    [2.030e+19, 8.262e-01], [3.045e+19, 8.118e-01], [4.061e+19, 8.011e-01],
    [5.076e+19, 7.921e-01], [6.092e+19, 7.841e-01], [7.107e+19, 7.769e-01],
    [8.123e+19, 7.702e-01], [9.353e+19, 7.626e-01], [1.015e+20, 7.579e-01],
    [2.031e+20, 7.094e-01], [3.047e+20, 6.712e-01], [4.062e+20, 6.378e-01],
    [5.078e+20, 6.074e-01], [6.094e+20, 5.791e-01], [7.109e+20, 5.524e-01],
    [8.125e+20, 5.271e-01], [9.355e+20, 4.977e-01], [1.016e+21, 4.793e-01],
    [2.031e+21, 2.837e-01], [3.047e+21, 1.526e-01], [4.063e+21, 7.774e-02],
    [5.078e+21, 3.952e-02], [6.094e+21, 2.093e-02], [7.110e+21, 1.199e-02],
    [8.125e+21, 7.666e-03], [9.355e+21, 5.333e-03], [1.016e+22, 4.666e-03],
])


def _log_log_interp(table, x_query):
    """1-D log-log interpolation / clipped extrapolation on a 2-column table."""
    log_x = np.log10(table[:, 0])
    log_y = np.log10(table[:, 1])
    log_xq = np.log10(np.maximum(x_query, table[0, 0]))
    # Extrapolate beyond the upper end with the slope of the last two points
    log_yq = np.interp(log_xq, log_x, log_y, right=log_y[-1])
    return 10.0 ** log_yq


def comp_SCO(N_CO, N_H2):
    """CO self-shielding factor.

    Port of RAMSES ``rtz/molecules_module.f90 :: comp_SCO``.
    Uses the Lee (1996) / Visser et al. (2009) lookup tables for the CO
    line self-shielding and the H2 continuum shielding of CO photodissociation.

    Parameters
    ----------
    N_CO : float or array-like
        CO column density [cm⁻²].
    N_H2 : float or array-like
        H2 column density [cm⁻²] (H2 molecules).

    Returns
    -------
    ndarray
        CO self-shielding factor ∈ (0, 1].
    """
    N_CO = np.asarray(N_CO, dtype=float)
    N_H2 = np.asarray(N_H2, dtype=float)
    s_CO = _log_log_interp(_SCO_TABLE,    N_CO)
    s_H2 = _log_log_interp(_SH2_CO_TABLE, N_H2)
    return s_CO * s_H2


# ═══════════════════════════════════════════════════════════════════════════════
#   Dust self-shielding of G0
# ═══════════════════════════════════════════════════════════════════════════════

def comp_G0_selfshield(tau_dust_FUV):
    """Dust self-shielding of the radiation field.

    The effective G0 seen by dust grains within the cell is reduced by the
    dust column of the cell itself, analogous to how ``comp_Sd`` shields H2.

    New implementation for RAMSES+CALIMA:

        G0_eff = G0 × exp(-τ_dust_FUV)
        τ_dust_FUV = Σ_i  κ_FUV,i × ρ_dust,i × dx

    In practice κ_FUV ≈ κ_LW (both in the 6–13.6 eV Habing band), so
    ``tau_dust_FUV`` can be the same quantity computed for ``comp_Sd_new``.

    Parameters
    ----------
    tau_dust_FUV : float or array-like
        Dust optical depth in the FUV / Habing band (dimensionless, ≥ 0).

    Returns
    -------
    ndarray
        G0 attenuation factor exp(-τ_dust_FUV) ∈ (0, 1].
    """
    return np.exp(-np.asarray(tau_dust_FUV, dtype=float))


# ═══════════════════════════════════════════════════════════════════════════════
#   κ_LW computation from Draine Q-tables (arbitrary grain size)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kappa_LW_draine(grain_sizes_um, composition='graphite',
                             E_min=E_LW_MIN, E_max=E_LW_MAX,
                             grain_density=None):
    """Compute LW-band mass opacity κ_LW [cm²/g] from Draine Q-tables.

    Reads the Draine & Lee (1984) grain efficiency tables (Gra_81 or suvSil_81)
    via the existing pyCALIMA ``dust_efficiencies`` reader, interpolates
    Q_abs(a, λ) at the requested grain sizes, averages over the Habing/LW band
    weighted by the Draine (1978) ISRF, then divides by grain mass.

    Parameters
    ----------
    grain_sizes_um : array-like
        Grain radii in microns.
    composition : {'graphite', 'silicate'}
        Grain composition.
    E_min, E_max : float
        Energy band limits in eV (default 6.0 – 13.6 eV, Habing band).
    grain_density : float or None
        Material density [g/cm³].  Defaults to 2.2 (graphite) or 3.3 (silicate).

    Returns
    -------
    kappa_LW : ndarray, shape same as ``grain_sizes_um``
        Mass opacity [cm²/g_dust] in the [E_min, E_max] band.
    """
    from pycalima.models.dust_radiation.dust_oppacity import dust_efficiencies

    grain_sizes_um = np.atleast_1d(np.asarray(grain_sizes_um, dtype=float))

    if grain_density is None:
        grain_density = 2.2 if composition == 'graphite' else 3.3

    if composition == 'graphite':
        table_file = str(_OPTICS_DIR / 'Gra_81')
    elif composition == 'silicate':
        table_file = str(_OPTICS_DIR / 'suvSil_81')
    else:
        raise ValueError(f"Unsupported composition: {composition!r}.")

    nwav, data, columns, _ = dust_efficiencies(table_file)

    # Sort grain sizes from table (dict keys are strings of grain size in µm)
    table_sizes_um = np.array(sorted(float(k) for k in data.keys()))
    log_a_tab = np.log10(table_sizes_um)

    # Wavelength grid from first entry [µm], descending → ascending
    first_key = min(data.keys(), key=float)
    wav_um_desc = data[first_key][:, 0]          # descending (long → short)
    sort_idx    = np.argsort(wav_um_desc)         # ascending wavelength
    wav_um      = wav_um_desc[sort_idx]

    # Energy grid [eV] and LW-band mask
    wav_cm  = wav_um * 1e-4
    E_eV    = h_cgs * c_cgs / (wav_cm * eV_to_erg)
    in_band = (E_eV >= E_min) & (E_eV <= E_max)
    if not in_band.any():
        raise ValueError(
            f"No wavelength points in [{E_min}, {E_max}] eV. "
            "Check Draine table wavelength coverage."
        )
    wav_nm_band = wav_um[in_band] * 1e3   # µm → nm

    # Draine (1978) ISRF photon flux [photons cm⁻² s⁻¹ nm⁻¹] (valid 91–200 nm)
    def _isrf(lnm):
        return 3.2028e13 * lnm**-3 - 5.1542e15 * lnm**-4 + 2.0546e17 * lnm**-5

    isrf_weight = np.maximum(_isrf(wav_nm_band), 0.0)
    isrf_norm   = np.trapezoid(isrf_weight, wav_nm_band)

    # Build 2-D Q_abs(ntable_sizes, nbandwav) array
    Q_abs_2d = np.zeros((len(table_sizes_um), in_band.sum()))
    for i, a_key in enumerate(sorted(data.keys(), key=float)):
        Q_col = data[a_key][:, columns.index('Q_abs')]
        Q_abs_2d[i, :] = Q_col[sort_idx][in_band]

    kappa_LW = np.zeros(len(grain_sizes_um))
    for j, a_um in enumerate(grain_sizes_um):
        a_um_clipped = np.clip(a_um, table_sizes_um.min(), table_sizes_um.max())
        log_a_j      = np.log10(a_um_clipped)

        # Log-log interpolation in grain size at each wavelength
        Q_abs_j = 10.0 ** np.array([
            np.interp(log_a_j, log_a_tab,
                      np.log10(np.maximum(Q_abs_2d[:, k], 1e-100)))
            for k in range(in_band.sum())
        ])

        a_cm     = a_um * 1e-4
        C_abs_j  = Q_abs_j * np.pi * a_cm**2            # cm²/grain
        C_abs_avg = (np.trapezoid(C_abs_j * isrf_weight, wav_nm_band) / isrf_norm
                     if isrf_norm > 0 else np.mean(C_abs_j))

        m_grain      = (4.0 / 3.0) * np.pi * grain_density * a_cm**3  # g
        kappa_LW[j]  = C_abs_avg / m_grain               # cm²/g_dust

    return kappa_LW


# ═══════════════════════════════════════════════════════════════════════════════
#   κ_LW from pre-computed pyCALIMA tables (default 6-bin configuration)
# ═══════════════════════════════════════════════════════════════════════════════

def _read_precomp_table(bin_id):
    """Read averaged_cross_section_<bin_id>.txt.

    Returns wavelengths_angstrom, C_abs_cm2, and the ISRF-averaged value.
    The stored C_abs values are cross-sections per representative grain [cm²].
    """
    fname = _MODEL_DIR / f"averaged_cross_section_{bin_id}.txt"
    if not fname.exists():
        raise FileNotFoundError(
            f"Pre-computed optical property file not found:\n  {fname}\n"
            "Run models/dust_radiation/export_dust_optical_properties.py first."
        )

    wavelengths = []
    cabs        = []
    isrf_avg    = None

    with open(fname) as f:
        lines = f.readlines()

    # Parse header: find NWAV, ISRF average, then data
    i = 0
    nwav = None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("# NWAV"):
            nwav = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("# ISRF_AVG_CROSS_SECTIONS_CM2"):
            vals = list(map(float, lines[i + 1].strip().split()))
            isrf_avg = {'C_abs': vals[0], 'C_sca': vals[1], 'C_rp': vals[2]}
            i += 2
        elif line.startswith("#"):
            i += 1
        elif line == "":
            i += 1
        else:
            # Data row: lambda[Å]  C_abs  C_sca  C_rp
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    cabs.append(float(parts[1]))
                except ValueError:
                    pass
            i += 1

    return np.array(wavelengths), np.array(cabs), isrf_avg


def compute_kappa_LW_precomp(bin_id, grain_density, a0_um,
                              E_min=E_LW_MIN, E_max=E_LW_MAX):
    """LW-band mass opacity κ_LW [cm²/g] from a pre-computed pyCALIMA table.

    Reads the averaged cross-section table for a given CALIMA dust bin,
    restricts to the LW/Habing band, performs a Draine (1978) ISRF-weighted
    average, and divides by the representative grain mass to get κ_LW.

    Parameters
    ----------
    bin_id : str
        CALIMA bin identifier, e.g. ``'DustBin_01'``.
    grain_density : float
        Material density of the grain [g/cm³] (2.2 graphite, 3.3 silicate).
    a0_um : float
        Representative grain radius [µm] (peak of the log-normal).
    E_min, E_max : float
        Energy band [eV].  Defaults to the Habing band (6–13.6 eV).

    Returns
    -------
    kappa_LW : float
        Mass opacity [cm²/g_dust].
    """
    wav_ang, C_abs, _ = _read_precomp_table(bin_id)

    # Convert wavelengths from Å to energy [eV]
    wav_cm = wav_ang * 1e-8
    E_eV   = h_cgs * c_cgs / (wav_cm * eV_to_erg)

    # Select LW band
    in_band = (E_eV >= E_min) & (E_eV <= E_max)
    if not in_band.any():
        raise ValueError(
            f"No data points in the band [{E_min}, {E_max}] eV for {bin_id}."
        )

    wav_nm_band  = wav_ang[in_band] * 0.1   # Å → nm
    C_abs_band   = C_abs[in_band]

    # ISRF photon-flux weighting (Draine 1978, valid 91–200 nm)
    def _isrf(lnm):
        return 3.2028e13 * lnm**-3 - 5.1542e15 * lnm**-4 + 2.0546e17 * lnm**-5

    w = np.maximum(_isrf(wav_nm_band), 0.0)
    norm = np.trapezoid(w, wav_nm_band)

    if norm > 0:
        C_abs_avg = np.trapezoid(C_abs_band * w, wav_nm_band) / norm
    else:
        C_abs_avg = np.mean(C_abs_band)

    # Representative grain mass [g]
    a0_cm   = a0_um * 1e-4
    m_grain = (4.0 / 3.0) * np.pi * grain_density * a0_cm**3

    return C_abs_avg / m_grain


# ═══════════════════════════════════════════════════════════════════════════════
#   Convenience: combined H2 shielding
# ═══════════════════════════════════════════════════════════════════════════════

def f_shd_old(N_HI, N_H2, Z=1.0):
    """Combined H2+dust shielding factor (old RAMSES formula).

    f_shd = comp_SH2(N_H2) × comp_Sd_old(N_HI, N_H2, Z)
    """
    return comp_SH2(N_H2) * comp_Sd_old(N_HI, N_H2, Z)


def f_shd_new(N_H2, tau_dust_LW):
    """Combined H2+dust shielding factor (new per-bin formula).

    f_shd = comp_SH2(N_H2) × comp_Sd_new(tau_dust_LW)
    """
    return comp_SH2(N_H2) * comp_Sd_new(tau_dust_LW)
